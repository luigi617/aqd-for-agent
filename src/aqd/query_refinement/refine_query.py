from typing import List, Tuple
from tqdm import tqdm
from pydantic import Field, BaseModel
from pydantic import BaseModel
from models.model import get_model
from aqd.schema import TreeNode



QUERY_REFINEMENT_PROMPT = """
Modify the current query based on the strategy. You should choose only one approach from the strategy to make the modification, and the new query must not be dramatically changed. Avoid using the same strategy as the previous refined query, if using the same strategy, do NOT have the same modification.

Current query:
{current_query}

Strategy:
{strategy}

Previously proposed refined queries, along with explanations of the improvements (do NOT reuse or echo these):
{previous_proposed_queries}

Output Format:
Returnyour response with the two elements 'strategy' and 'query'. The 'strategy' value contains a few sentences interpreting the strategy used to modify the response. The 'query' value contains the new query.
"""


class QueryRefinementSchema(BaseModel):
    strategy: str = Field(description="Few sentences interpreting the strategy used to modify the response.")
    query: str = Field(description="Refined query.")



def get_previous_proposed_queries(proposed_queries: List[Tuple[str, str]]) -> str:
    previous_proposed_queries_str = ""
    for proposed_query in proposed_queries:
        previous_proposed_queries_str += (
            f"Query: {proposed_query[0]}\n"
            f"Strategy: {proposed_query[1]}\n"
            "\n"
        )
    if not previous_proposed_queries_str:
        previous_proposed_queries_str = "[No refined queries proposed]"
    return previous_proposed_queries_str


def refine_query(
        tree_node: TreeNode,
        model_path: str,
        query_refinement_strategy: str,
        n_refinement: int = 3,
    ) -> List[Tuple[str, str]]:

    refine_query_chain = get_model(model_path, QueryRefinementSchema)
    
    query_node = tree_node.value
    query = query_node.query

    proposed_queries = []
    for _ in range(n_refinement):
        previous_proposed_queries = get_previous_proposed_queries(proposed_queries)
        refinement = refine_query_chain(
            QUERY_REFINEMENT_PROMPT.format(
                current_query=query,
                strategy=query_refinement_strategy,
                previous_proposed_queries=previous_proposed_queries,
            )
        )
        proposed_queries.append((refinement["query"], refinement["strategy"]))

    return proposed_queries
