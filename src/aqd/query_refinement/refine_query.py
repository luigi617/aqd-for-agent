from typing import List, Tuple
from tqdm import tqdm
from pydantic import Field, BaseModel
from pydantic import BaseModel
from src.aqd.budget import Budget
from src.models.model import get_model
from src.aqd.schema import TreeNode

# Stretegy instructions taken from https://arxiv.org/pdf/2310.13345

QUERY_REFINEMENT_PROMPT = """
Given the current query, generate a new query by the guidance of the perturbation strategy while keeping the meaning of the original query.

Current query:
{current_query}

Perturbation strategy:
{strategy}

Output Format:
Return the new query.
"""


class QueryRefinementSchema(BaseModel):
    query: str = Field(description="Refined query.")


def refine_query(
        tree_node: TreeNode,
        model_path: str,
        budget: Budget,
        query_refinement_strategies: list
    ) -> List[Tuple[str, str]]:

    refine_query_chain = get_model(model_path, QueryRefinementSchema, budget=budget)
    
    query_node = tree_node.value
    query = query_node.query

    proposed_queries = []
    for strategy in query_refinement_strategies:
        refinement = refine_query_chain(
            QUERY_REFINEMENT_PROMPT.format(
                current_query=query,
                strategy=strategy,
            )
        )
        proposed_queries.append((refinement["query"], strategy))

    return proposed_queries
