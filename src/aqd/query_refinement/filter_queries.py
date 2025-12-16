from pydantic import Field, BaseModel
from pydantic import BaseModel
from src.aqd.budget import Budget
from src.aqd.schema import QueryNode
from src.models.model import get_model


FILTER_QUERY_PROMPT = """\
You are a filter that decides whether the user's query satisfies the filtering policy.

Query:
{query}

Filtering policy:
{filtering_policy}

Output Format:
Return the reasoning that explains whether the query satisfies the filtering policy, followed by the boolean value True if the query satisfies the filtering policy, and False otherwise. Finally, provide a confidence score between 0 and 1 indicating the confidence in your filtering decision.
"""

class FilteredQuerySchema(BaseModel):
    reasoning: str = Field(description="reasoning behind the query satisfing or not satisfying the filtering policy.")
    satisfy_filter: bool = Field(description="True if the query satisfies the filtering policy, False otherwise.")
    confidence_score: float = Field(description="A confidence score between 0 and 1 indicating the confidence in the filtering decision.")

def filter_queries(
        query_node: QueryNode,
        model_path: str,
        budget: Budget,
        filtering_policy: str,
    ) -> tuple[bool, str]:
    """
    Filters queries based on filtering policy definition.
    """
    if query_node.filtering_reason != "":
        return not query_node.is_filtered_out, query_node.filtering_reason, query_node.filtering_confidence_score
    filter_query_chain = get_model(model_path, FilteredQuerySchema, retries=3, budget=budget)

    query = query_node.query
    decision = filter_query_chain(
        FILTER_QUERY_PROMPT.format(
            query=query,
            filtering_policy=filtering_policy
        )
    )
    
    return decision['satisfy_filter'], decision['reasoning'], decision['confidence_score']