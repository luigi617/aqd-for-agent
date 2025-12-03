
from typing import List, Tuple
from pydantic import Field, BaseModel
from src.aqd.budget import Budget
from src.aqd.schema import PromptNode, QueryNode
from src.models.model import get_model



QUERY_FILTER_FEEDBACK_PROMPT = """
You are an expert assistant participating in an iterative prompt-optimization process for the e-commerce domain.
Your role is to analyze a set of user-query examples and their corresponding filtering results, then provide feedback to help refine a prompt that will be used to generate future user queries.

The prompt you aim to refine will later be combined with a product name and several random attribute names to form a complete prompt. This complete prompt will be used exclusively for generating user queries that must comply with a given query-filtering policy.

Below, you will find generated queries that have been filtered out according to the query filtering policy.
Your task is to explain 1) why the current prompt may have produced queries that violated the filtering policy and 2) propose concrete improvements to the prompt so that future generated queries will not be filtered out.

Current prompt set:
{current_prompts}

Filtered out queries:
{filtered_out_queries}

Query filtering policy:
{filtering_policy}

Output Format:
Return first a clear explaination of why the current prompt resulted in queries that failed to meet the filtering policy. Then, provice specific, actionable suggestions for refining or rewording the prompt to prevent future generated queries from being filtered out.
"""



class QueryFilterFeedbackSchema(BaseModel):
    
    reasoning: str = Field(description="Explanation of why the current prompt produced queries not satisfying query filtering policy.")
    
    improvement_suggestions: str = Field(
        description="Guidance on how to refine or reword the prompt to prevent queries to be filtered out."
    )

def get_filtered_out_queries(query_nodes: List[QueryNode]):
    filtered_out_queries = ""
    
    for query_node in query_nodes:
        if query_node.is_filtered_out:
            filtered_out_queries += (
                f"Query: {query_node.query}\n"
                f"Reason for violating query filtering policy: {query_node.filtering_reason}\n"
                "\n"
            )
 
    if not filtered_out_queries:
        filtered_out_queries = "[No filtered-out queries generated]"
    return filtered_out_queries


def query_filter_feedback(
    prompt_node: PromptNode,
    model_path: str,
    budget: Budget,
    filtering_policy: str,
) -> Tuple[str, List[str], List[str]]:
    if prompt_node.query_filter_reason != "" and prompt_node.query_filter_improvement_suggestions != "":
        return prompt_node.query_filter_reason, prompt_node.query_filter_improvement_suggestions
    
    query_filtering_chain = get_model(model_path, QueryFilterFeedbackSchema, budget=budget)
    prompt = prompt_node.prompt

    filtered_out_queries = get_filtered_out_queries(prompt_node.queries)

    query_filtering_feedback = query_filtering_chain(
        QUERY_FILTER_FEEDBACK_PROMPT.format(
            current_prompts=prompt, 
            filtered_out_queries=filtered_out_queries,
            filtering_policy=filtering_policy
        )
    )

    return query_filtering_feedback["reasoning"], query_filtering_feedback["improvement_suggestions"]