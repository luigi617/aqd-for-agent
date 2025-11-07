
from typing import List, Tuple
from pydantic import Field, BaseModel
from aqd.schema import PromptNode, QueryNode
from models.model import get_model



UNPREFERRED_RESPONSE_FEEDBACK_PROMPT = """
You are an expert assistant in an iterative prompt-optimization process for the e-commerce domain.
Your role is to refine and create prompts that guide the generation of user queries. These prompts are used exclusively for generating user queries, and random attribute names may be provided as inputs.

Some generated user queries were answered by a large language model in a non-preferred way, as defined below.
Your task is to analyze and explain why the current prompt likely caused user queries that resulted in non-preferred responses. Additionally, propose concrete improvements to the prompt to ensure that produce user queries are more likely to elicit preferred responses.

Current prompt set:
{current_prompts}

User queries with non-preferred responses and reasons:
{unpreferred_responses}

Non-preferred criteria:
{unpreferred_criteria}

Output Format:
Return first a string explaining how the current prompt produces queries that receive non-preferred responses. Then, provide a contributing prompt pattern, a list of specific patterns, phrases, or logic in the prompt that likely caused the queries to generate non-preferred responses. Finally, include a list of improvement suggestions: guidance on how to refine or reword the prompt to generate user queries that are more likely to produce preferred responses.
"""



class UnpreferredResponseFeedbackSchema(BaseModel):
    
    reasoning: str = Field(description="Explanation of how the current prompt produces queries that are answered in a non-preferred way.")
    contributing_prompt_patterns: List[str] = Field(
        default_factory=list,
        description="Phrases/rules in the prompt that generate queries that promote non-preferred responses."
    )
    improvement_suggestions: List[str] = Field(
        default=list,
        description="Guidance on how to refine the prompt to generate user queries that are more likely to produce unpreferred response."
    )

def get_unpreferred_criteria_str(unpreferred_definition):
    unpreferred_criteria = []
    for criteria in unpreferred_definition.values():
        unpreferred_criteria.extend(criteria)
    return "\n".join([f"- {c}" for c in unpreferred_criteria])

def get_unpreferred_response_str(query_nodes: List[QueryNode], unpreferred_definition: dict):
    unpreferred_builder = ""
    for query_node in query_nodes:
        if not query_node.is_filtered_out:
            continue

        query = query_node.query
        response = query_node.response
        criteria_chosen = query_node.evaluation
        eval_reason = query_node.evaluation_reason
    
        is_unpreferred = False
        for category, _ in criteria_chosen:
            if category in unpreferred_definition:
                is_unpreferred = True
                break

        if is_unpreferred:
            unpreferred_builder += (
                f"Query: {query}\n"
                f"Response: {response}\n"
                f"Reason for unpreferred: {eval_reason}\n\n"
            )

    if not unpreferred_builder:
        unpreferred_builder = "[No unpreferred responses were generated]"
    return unpreferred_builder


def unpreferred_response_feedback(
    prompt_node: PromptNode,
    model_path: str,
    unpreferred_definition: dict,
) -> Tuple[str, List[str], List[str]]:
    
    unpreferred_response_chain = get_model(model_path, UnpreferredResponseFeedbackSchema)
    prompt = prompt_node.prompt

    unpreferred_criteria = get_unpreferred_criteria_str(unpreferred_definition)
    unpreferred_responses = get_unpreferred_response_str(prompt_node, unpreferred_definition)
    unpreferred_response_feedback = unpreferred_response_chain(
        UNPREFERRED_RESPONSE_FEEDBACK_PROMPT.format(
            current_prompts=prompt, 
            unpreferred_responses=unpreferred_responses,
            unpreferred_criteria=unpreferred_criteria
        )
    )

    return unpreferred_response_feedback["reasoning"], unpreferred_response_feedback["contributing_prompt_patterns"], unpreferred_response_feedback["improvement_suggestions"]