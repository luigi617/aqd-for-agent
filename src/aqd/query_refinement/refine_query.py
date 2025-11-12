from typing import List, Tuple
from tqdm import tqdm
from pydantic import Field, BaseModel
from pydantic import BaseModel
from models.model import get_model
from aqd.schema import TreeNode

# Stretegy instructions taken from https://arxiv.org/pdf/2310.13345
QUERY_REFINEMENT_STRATEGIES = [
    "Choose at most two words in the sentence, and change them so that they have typos.",
    "Change at most two letters in the sentence.",
    "Add at most two extraneous characters to the end of the sentence.",
    "Replace at most two words in the sentence with synonyms.",
    "Choose at most two words in the sentence that do not contribute to the meaning of the sentence and delete them.",
    "Add at most two semantically neutral words to the sentence.",
    "Add a randomly generated short meaningless handle after the sentence, such as @fasuv3.",
    "Paraphrase the sentence.",
    "Change the syntactic structure of the sentence.",
]

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
    ) -> List[Tuple[str, str]]:

    refine_query_chain = get_model(model_path, QueryRefinementSchema)
    
    query_node = tree_node.value
    query = query_node.query

    proposed_queries = []
    for perburtation_strategies in QUERY_REFINEMENT_STRATEGIES:
        refinement = refine_query_chain(
            QUERY_REFINEMENT_PROMPT.format(
                current_query=query,
                strategy=perburtation_strategies,
            )
        )
        proposed_queries.append((refinement["query"], perburtation_strategies))

    return proposed_queries
