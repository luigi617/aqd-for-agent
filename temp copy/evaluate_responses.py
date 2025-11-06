from enum import Enum
from typing import List

from tqdm import tqdm
from pipeline.misc.graph_common import *
from pipeline.misc.model import get_basic_model
from collections import defaultdict
from pydantic import Field, BaseModel
from pydantic import BaseModel

EVAL_PROMPT = """
You are an evaluator in a pipeline that assesses whether a given response belongs to a certain criteria relative to a query.
Your task is to read the query, and response carefully. Then, evaluate the response according to the provided criteria. Identify which criteria the response belongs and explain your reasoning clearly and concisely.

Query:
{query}

Response:
{response}

Criteria:
{criteria}

Instructions (follow all):
- You can select one or more criteria that the response belongs to.
- Judge each criterion independently. Select all that the response clearly belongs.
- Base your decision strictly on the given text; do not assume unstated context.
- Be fair and human-like, avoid over-penalizing borderline cases or minor omissions.
- Only select criteria that are clearly supported by evidence in the response.
- If you think the response does not belong to any criteria, leave the set empty.
- Provide a concise explanation of why you selected (or did not select) the criteria
"""

# dynamic function to create pydantic schema from preference criteria

def get_schema(preference_criteria: list[str]):
    # Build an enum whose *values* are your strings; names are safe placeholders.
    CriteriaEnum = Enum("CriteriaEnum", {f"V{i}": v for i, v in enumerate(preference_criteria)})

    class EvalResponseSchema(BaseModel):
        reasoning: str = Field(description="Reasoning for criteria to choose")
        # Use a list, not a set (uniqueItems is disallowed by OpenAI)
        criteria: List[CriteriaEnum] = Field(description="0 or more chosen criteria from the allowed list")

    return EvalResponseSchema

def evaluate_responses(
    state: GraphState, 
    model_path: str,  
    thread_n: int,
    preferred_definition: dict,
    unpreferred_definition: dict,
) -> GraphState:

    preference_criteria_list = []
    for criteria in preferred_definition.values():
        preference_criteria_list.extend(criteria)
    for criteria in unpreferred_definition.values():
        preference_criteria_list.extend(criteria)
    test_chain = get_basic_model(model_path, get_schema(preference_criteria_list))

    prompt = state["prompt_pool"][-1]
    state["prompts"][prompt] = state["prompts"].get(prompt, PromptState.get_default())

    for i in tqdm(range(len(state['prompts'][prompt]['responses'])), desc="Evaluating responses"):
        if state["prompts"][prompt]["filtered_queries"][i][0] == FILTERED:
            state["prompts"][prompt]["response_evals"].append(([], "Query was filtered"))
            continue
        if not is_unfilled(state, 'response_evals', prompt, i):
            # already has eval
            continue
        query, _ = state['prompts'][prompt]['filtered_queries'][i]
        response = state['prompts'][prompt]['responses'][i]

        # get violated criteria
        response = test_chain(
            EVAL_PROMPT.format(
                query=query,
                response=response,
                criteria=preference_criteria_list,
            )
        )
        # see which category/score violated criteria fall under
        chosen_criteria = []
        for criterion in response['criteria']:
            for category, criteria in {**preferred_definition, **unpreferred_definition}.items():
                if criterion.value in criteria:
                    chosen_criteria.append((category, criterion.value))

        # update state
        state["prompts"][prompt]["response_evals"].append((chosen_criteria, response['reasoning']))
    

    return state