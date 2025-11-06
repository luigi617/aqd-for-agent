from collections import defaultdict

from tqdm import tqdm
from pipeline.misc.graph_common import (
    GraphState,
    PromptState,
    get_attr_str,
    is_unfilled,
)
from pipeline.misc.model import get_basic_model
from pydantic import Field, BaseModel
from random import randint, choice

QUERY_GENERATION_PROMPT = """
{template}

Product:
{product_name}

Attributes:
{attributes}

Previously used queries (avoid similar intent/wording):
{previous_queries}

Instructions (follow all):
- Do NOT include overly specific product identifiers (e.g., model numbers, SKUs, GTINs).
- Do NOT repeat or closely rephrase previous queries.
- Query must be limited to at most 2 of the given attributes.
"""

class QueryResponseSchema(BaseModel):
    chain_of_thought: str = Field(description="Reasoning about how the query was formed using the template and attributes.")
    query: str = Field(description="The final generated query string.")


def get_random_attributes(product):
    
    attributes = list(product.keys())
    # starting attributes (must be shared across all products)

    attrs = []
    n_attrs = randint(1, 3)
    
    attempts = 0
    max_attempts = max(8, n_attrs * 6)
    while n_attrs > 0 and attempts < max_attempts:
        attempts += 1
        attr = choice(attributes)

        # skip duplicates
        if attr in attrs:
            continue

        # validate value: must exist and not be overly long
        val = product.get(attr)
        if not val:
            continue

        attrs.append(attr)
        n_attrs -= 1  
    return attrs  

def generate_queries(
        state: GraphState, 
        model_path: str, 
        products: list[dict],
        thread_n: int,
        n_queries: int = 5
    ) -> GraphState:
    """
    Generate queries based the prompt pool with product attributes.
    Returns updates to state with new queries, products, and attributes.
    Each prompt will have n_queries total after this function runs.
    """
    query_chain = get_basic_model(model_path, QueryResponseSchema)
    prompt = state["prompt_pool"][-1]

    state["prompts"][prompt] = state["prompts"].get(prompt, PromptState.get_default())
    
    previous = state["prompts"].get(prompt, {"template_queries": []})["template_queries"].copy()
    for i in tqdm(range(n_queries), desc="Generating queries"):
        if not is_unfilled(state, 'template_queries', prompt, i):
            continue
        # get random product to use as information for filling
        product = choice(products)
        
        attributes = get_random_attributes(product)

        # join attributes
        attributes_str = ", ".join(attributes)
        

        # get new query and add to previous
        previous_queries = '\n'.join(previous) if previous else '[No previous queries — this is the first query]'

        new_query = query_chain(
            QUERY_GENERATION_PROMPT.format(
                template=prompt,
                product_name=product.get("product_name"),
                attributes=attributes_str,
                previous_queries=previous_queries,
            )
        )['query']
        previous.append(new_query)

        # update state
        state["prompts"][prompt]["template_queries"].append(new_query)
        state["prompts"][prompt]["products"].append(product)
        state["prompts"][prompt]["attributes"].append(attributes)

    return state