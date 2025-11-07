from enum import Enum
from typing import List, Tuple

from pydantic import Field, BaseModel
from pydantic import BaseModel
from aqd.schema import QueryNode
from models.model import get_model

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

Output format:
Return the reasoning for all chosen criteria, followed by a list of criteria whose values meet the selection condition.
"""

# dynamic function to create pydantic schema from preference criteria

def get_schema(preference_criteria_map: dict):
    # Build an enum whose *values* are your strings; names are safe placeholders.
    CriteriaEnum = Enum("CriteriaEnum", {k: v for k, v in preference_criteria_map.items()})

    class EvalResponseSchema(BaseModel):
        reasoning: str = Field(description="Reasoning for criteria to choose")
        criteria: List[CriteriaEnum] = Field(
            default_factory=list,
            description="0 or more chosen criteria from the allowed list"
        )

    return EvalResponseSchema

def get_preference_criteria_map(preferred_definition, unpreferred_definition):
    i = 0
    preference_criteria_map = {}
    for criteria in preferred_definition.values():
        for criterion in criteria:
            preference_criteria_map[f"V{i}"] = criterion
            i += 1
    for criteria in unpreferred_definition.values():
        for criterion in criteria:
            preference_criteria_map[f"V{i}"] = criterion
            i += 1
    return preference_criteria_map

def get_preference_criteria_str(preference_criteria_map):
    preference_criteria_str = ""
    for preference_key, preference_value in preference_criteria_map.items():
        preference_criteria_str += f"key: {preference_key}, value: {preference_value}\n"
    return preference_criteria_str
    
def evaluate_responses(
    query_node: QueryNode,
    model_path: str,  
    preferred_definition: dict,
    unpreferred_definition: dict,
) -> Tuple[List[Tuple[str, str]], str]:
    """
    Evaluate response
    """

    preference_criteria_map = get_preference_criteria_map(preferred_definition, unpreferred_definition)
    preference_criteria_str = get_preference_criteria_str(preference_criteria_map)
    evaluate_response_chain = get_model(model_path, get_schema(preference_criteria_map))

    query = query_node.query
    response = query_node.response
        
    # get violated criteria
    evaluation = evaluate_response_chain(
        EVAL_PROMPT.format(
            query=query,
            response=response,
            criteria=preference_criteria_str,
        )
    )
    # see which category/score violated criteria fall under
    chosen_criteria = []
    for criterion in evaluation['criteria']:
        for category, criteria in {**preferred_definition, **unpreferred_definition}.items():
            if criterion.value in criteria:
                chosen_criteria.append((category, criterion.value))

    return chosen_criteria, evaluation["reasoning"]