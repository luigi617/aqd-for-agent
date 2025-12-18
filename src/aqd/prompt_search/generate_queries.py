from typing import List, Tuple
from tqdm import tqdm
from pydantic import Field, BaseModel
from random import randint, choice

from src.aqd.budget import Budget
from src.models.model import get_model
from src.aqd.schema import PromptNode

QUERY_GENERATION_PROMPT = """
{template}

Product:
{product_name}

Attributes:
{attributes}

Instructions (follow all):
- Do NOT include overly specific product identifiers (e.g., model numbers, SKUs, GTINs).
- Query must be limited to at most 2 of the given attributes.

Output Format:
Return the final generated query string.
"""

QUERY_GENERATION_PROMPT_BASELINE = """
{template}

Product:
{product_name}

Attributes:
{attributes}

Instructions (follow all):
- Do NOT include overly specific product identifiers (e.g., model numbers, SKUs, GTINs).
- Query must be limited to at most 2 of the given attributes.

Output Format:
Return the final generated list of query strings.
"""

class QueryResponseSchema(BaseModel):
    query: str = Field(description="The final generated query string.")

class BaselineQueryResponseSchema(BaseModel):
    queries: List[str] = Field(
        min_length=30, 
        max_length=30, 
        description="A list of exactly 30 different generated queries about the product and its attribute"
    )

def get_random_attributes(product):
    
    attributes = list(product.keys())

    attrs = []
    n_attrs = randint(1, 3)
    
    attempts = 0
    max_attempts = max(8, n_attrs * 6)
    while n_attrs > 0 and attempts < max_attempts:
        attempts += 1
        attr = choice(attributes)
        
        if attr in attrs:
            continue

        val = product.get(attr)
        if not val:
            continue

        attrs.append(attr)
        n_attrs -= 1  
    return attrs 


def generate_queries(
        prompt_node: PromptNode,
        model_path: str, 
        budget: Budget,
        products: list[dict],
        product_idx: int,
        n_queries: int = 5,
    ) -> List[Tuple[str, dict, list]]:
    """
    Generate queries based the prompt pool with product attributes.
    Each prompt will have n_queries total after this function runs.
    """
    query_chain = get_model(model_path, QueryResponseSchema, budget=budget)
    
    prompt = prompt_node.prompt
    
    generated_queries = []
    for _ in range(n_queries):
        product = products[product_idx % len(products)]
        attributes = get_random_attributes(product)
        attributes_str = ", ".join(attributes)

        new_query = query_chain(
            QUERY_GENERATION_PROMPT.format(
                template=prompt,
                product_name=product.get("product_name"),
                attributes=attributes_str,
            )
        )['query']
        
        generated_queries.append((new_query, product, attributes, product_idx))
        product_idx += 1

    return generated_queries

def generate_queries_baseline(
        prompt_node: PromptNode,
        model_path: str, 
        budget: Budget,
        product: dict,
    ) -> List[Tuple[str, dict, list]]:
    """
    Generate queries based the prompt pool with product attributes.
    Each prompt will have n_queries total after this function runs.
    """
    query_chain = get_model(model_path, BaselineQueryResponseSchema, budget=budget)
    
    prompt = prompt_node.prompt
    
    generated_queries = []
    attributes = product['selected_attributes']
    attributes_str = ", ".join(attributes)

    new_query = query_chain(
        QUERY_GENERATION_PROMPT.format(
            template=prompt,
            product_name=product.get("product_name"),
            attributes=attributes_str,
        )
    )['queries']
    
    generated_queries.append((new_query, product, attributes, product_idx))
    product_idx += 1

    return generated_queries