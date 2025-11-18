
from typing import List, Tuple
from pydantic import Field, BaseModel
from aqd.schema import PromptNode, QueryNode
from models.model import get_model



QUERY_FILTER_FEEDBACK_PROMPT = """
You are an expert assistant participating in an iterative prompt optimization process within the e-commerce domain.
Your role is to refine and create prompts that guide the generation of user queries. These prompts are used exclusively for generating user queries, and random attribute names will be provided for this purpose.

Some of the generated user queries have been filtered out based on a defined query filtering policy.
Your task is to analyze and explain why the current prompt might have led to those filtered-out queries, and to propose concrete improvements to avoid queries being filtered out.

Current prompt set:
{current_prompts}

Filtered out queries:
{filtered_out_queries}

Query filtering policy:
{filtering_policy}

Output Format:
Return first a string explaining why the current prompt produced queries that did not satisfy the query filtering policy. Then, return an improvement suggestions: guidance on how to refine or reword the prompt to prevent the queries from being filtered out.
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
    filtering_policy: str,
) -> Tuple[str, List[str], List[str]]:
    
    query_filtering_chain = get_model(model_path, QueryFilterFeedbackSchema)
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