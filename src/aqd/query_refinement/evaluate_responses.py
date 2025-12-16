from collections import defaultdict
from enum import Enum
from typing import List, Tuple

from pydantic import Field, BaseModel, field_validator
from src.aqd.budget import Budget
from src.aqd.schema import QueryNode
from src.models.model import get_model

EVAL_PROMPT = """
You are an evaluator in a pipeline that assesses whether a given response satisfies a certain criteria relative to a query.
Your task is to read the query, and response carefully. Then, evaluate the response according to the provided criteria. Base your decision strictly on the given query and response, do not assume unstated context. Identify and select all the  criteria the response satisfies and explain your reasoning clearly and concisely.

Query:
{query}

Response:
{response}

Criteria:
{criteria}

Instructions (follow all):
- Judge each criterion independently. Select all that the response clearly belongs.
- Be fair and human-like, avoid over-penalizing borderline cases or minor omissions.
- Only select criteria that are clearly supported by evidence in the response.
- If you think the response does not belong to any criteria, leave the set empty.

Output format:
Return the reasoning for all selected criteria, then list the keys of the criteria whose values apply to the response
"""

MERGE_REASONING_PROMPT = """
You are an adjudicator that merges multiple evaluation explanations into one final explanation.

You are given a set of FINAL_CRITERIA that represents the final criteria chosen and a list of INDIVIDUAL_EXPLANATIONS from different evaluation runs or different LLM judges. Your task is to produce ONE unified explanation that justify only the criteria chosen in FINAL_CRITERIA

FINAL_CRITERIA:
{final_criteria}

INDIVIDUAL_EXPLANATIONS:
{explanations}

Instructions (follow all):
- Do NOT justify or mention criteria that are not chosen.
- Synthesize evidence from the individual explanations, but avoid contradictions.
- The final explanation should be concise, coherent, and human-like.
- Do NOT mention the existence of multiple judges or that you are merging explanations.

Output:
A single paragraph explaining why the response satisfies the final chosen criteria.
"""


# dynamic function to create pydantic schema from preference criteria
def get_schema(preference_criteria_map: dict[str, str]):
    allowed = list(preference_criteria_map.keys())
    class EvalResponseSchema(BaseModel):
        reasoning: str = Field(description="Reasoning for criteria to choose")
        criteria: List[str] = Field(
            default_factory=list,
            description=f"0 or more chosen criteria. Allowed values: {allowed}"
        )

        # Accept single string, null, or list
        @field_validator("criteria", mode="before")
        @classmethod
        def coerce_to_list(cls, v):
            if v is None:
                return []
            if isinstance(v, str):
                return [v]
            return v

        # Normalize case/whitespace, verify membership, and dedupe while preserving order
        @field_validator("criteria")
        @classmethod
        def normalize_and_validate(cls, selected_criteria):
            out = set()
            for criteria_key in selected_criteria:
                if criteria_key not in preference_criteria_map:
                    raise ValueError(
                        f"Unknown criterion: {criteria_key}. Allowed: {allowed}"
                    )
                out.add(preference_criteria_map[criteria_key])
            return list(out)

    return EvalResponseSchema

class EvalReasonMergingSchema(BaseModel):
    reasoning: str = Field(description="Unified reasoning for chosen criteria")

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

def get_preference_criteria_to_key_map(preference_criteria_map):
    return {v:k for k, v in preference_criteria_map.items()}

def get_preference_criteria_str(preference_criteria_map):
    preference_criteria_str = ""
    for preference_key, preference_value in preference_criteria_map.items():
        preference_criteria_str += f"key: {preference_key}, value: {preference_value}\n"
    return preference_criteria_str

def get_majority(n_evaluator, n_evaluation):
    return (n_evaluator * n_evaluation) // 2 + 1

def evaluate_responses(
    query_node: QueryNode,
    evaluation_model_paths: List[str],
    generative_model_path: str,
    budget: Budget,
    preferred_definition: dict,
    unpreferred_definition: dict,
    n_evaluation: int = 1,
) -> Tuple[List[Tuple[str, str]], str]:
    """
    Evaluate response
    """
    if query_node.evaluation and query_node.evaluation_reason is not None and query_node.individual_evaluations:
        return query_node.evaluation, query_node.evaluation_reason, query_node.individual_evaluations
    
    preference_criteria_map = get_preference_criteria_map(preferred_definition, unpreferred_definition)
    preference_criteria_to_key_map = get_preference_criteria_to_key_map(preference_criteria_map)
    preference_criteria_str = get_preference_criteria_str(preference_criteria_map)
    
    query = query_node.query
    response = query_node.response

    evaluations = []
    reason_merging_chain = get_model(generative_model_path, EvalReasonMergingSchema, retries=3, budget=budget)
    for model_path in evaluation_model_paths:
        evaluate_response_chain = get_model(model_path, get_schema(preference_criteria_map), retries=5, budget=budget, timeout=300)
        
        # get violated criteria
        for _ in range(n_evaluation):
            evaluation = evaluate_response_chain(
                EVAL_PROMPT.format(
                    query=query,
                    response=response,
                    criteria=preference_criteria_str,
                )
            )
            evaluations.append(evaluation)

    criteria_count = defaultdict(int)
    for evaluation in evaluations:
        for criterion in evaluation['criteria']:
            criteria_count[criterion] += 1
    preference_criterion_to_category_map = {}
    for category, criteria in {**preferred_definition, **unpreferred_definition}.items():
        for criterion in criteria:
            preference_criterion_to_category_map[criterion] = category
    
    chosen_criteria = []
    threshold = get_majority(len(evaluation_model_paths), n_evaluation)
    for criterion, count in criteria_count.items():
        if count >= threshold:
            category = preference_criterion_to_category_map[criterion]
            chosen_criteria.append((category, criterion))
    
    all_reasoning = "\n".join([f"- {evaluation['reasoning']}" for evaluation in evaluations])
    final_selected_criteria_str = "\n".join([f"key = {preference_criteria_to_key_map[criterion]}, value = {criterion}" for _, criterion in chosen_criteria])

    reasoning = reason_merging_chain(
        MERGE_REASONING_PROMPT.format(
            final_criteria=final_selected_criteria_str,
            explanations=all_reasoning,
        )
    )["reasoning"]


    return chosen_criteria, reasoning, evaluations