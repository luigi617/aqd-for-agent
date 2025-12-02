
from typing import List, Tuple
from pydantic import Field, BaseModel
from src.aqd.budget import Budget
from src.aqd.schema import PromptNode, QueryNode
from src.models.model import get_model



UNPREFERRED_RESPONSE_FEEDBACK_PROMPT = """
You are an expert assistant participating in an iterative prompt-optimization process for the e-commerce domain.
Your role is to analyze a set of user-query and response examples and their corresponding evaluation results, then provide feedback to help refine a prompt that will be used to generate future user query.

The prompt you aim to refine will later be combined with a product name and several random attribute names to form a complete prompt. This complete prompt will be used exclusively for generating a user query that must comply with a given query-filtering policy.

Below, you will find generated user queries that resulted in non-preferred responses from an AI agent, based on the defined non-preferred criteria.
Your task is to analyze and explain why the current prompt likely produced user queries that elicited these non-preferred responses from the agent, in order to clarify why the agent failed to respond in a preferred way. Additionally, propose concrete improvements to the prompt to ensure it generates user queries that, when presented to the same AI agent, are more likely to trigger non-preferred responses.

Current prompt set:
{current_prompts}

User queries with non-preferred responses and reasons:
{unpreferred_responses}

Non-preferred criteria:
{unpreferred_criteria}

Output Format:
Return first a clear explaination on how the current prompt produces query that receive non-preferred responses. Then, provide specific, actionable improvement suggestions on how to refine or reword the prompt to generate user queries that are more likely to produce unpreferred responses.
"""



class UnpreferredResponseFeedbackSchema(BaseModel):
    
    reasoning: str = Field(description="Explanation of how the current prompt produces queries that are answered in a non-preferred way.")
    
    improvement_suggestions: str = Field(
        description="Guidance on how to refine the prompt to generate user queries that are more likely to produce unpreferred response."
    )

def get_unpreferred_criteria_str(unpreferred_definition):
    unpreferred_criteria = []
    for criteria in unpreferred_definition.values():
        unpreferred_criteria.extend(criteria)
    return "\n".join([f"- {c}" for c in unpreferred_criteria])

def get_unpreferred_response_str(prompt_node: PromptNode):
    unpreferred_builder = ""
    for query_node in prompt_node.queries:
        if not query_node.is_filtered_out:
            continue

        query = query_node.query
        response = query_node.response
        eval_reason = query_node.evaluation_reason
    
        is_unpreferred = query_node.is_unpreferred

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
    budget: Budget,
    unpreferred_definition: dict,
) -> Tuple[str, List[str], List[str]]:
    if prompt_node.unpreferred_response_reason != "" and prompt_node.unpreferred_response_improvement_suggestions != "":
        return prompt_node.unpreferred_response_reason, prompt_node.unpreferred_response_improvement_suggestions
    
    unpreferred_response_chain = get_model(model_path, UnpreferredResponseFeedbackSchema, budget=budget)
    prompt = prompt_node.prompt

    unpreferred_criteria = get_unpreferred_criteria_str(unpreferred_definition)
    unpreferred_responses = get_unpreferred_response_str(prompt_node)
    unpreferred_response_feedback = unpreferred_response_chain(
        UNPREFERRED_RESPONSE_FEEDBACK_PROMPT.format(
            current_prompts=prompt, 
            unpreferred_responses=unpreferred_responses,
            unpreferred_criteria=unpreferred_criteria
        )
    )

    return unpreferred_response_feedback["reasoning"], unpreferred_response_feedback["improvement_suggestions"]