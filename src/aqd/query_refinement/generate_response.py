from pydantic import Field, BaseModel
from pydantic import BaseModel
from src.aqd.budget import Budget
from src.aqd.schema import QueryNode
from src.models.model import get_model

RESPONSE_GEN_PROMPT = """
Provide a direct answer to the user's query based on the provided attributes.

Product attributes:
{attributes}

User query:
{query}

Instructions (follow all)
- Do NOT mention field names or say “according to the attributes”.
- Do not hallucinate any information not present.

Output Format:
Return the generated response to the query.
"""


class ResponseSchema(BaseModel):
    response: str = Field(description="A generated response to the query.")

def get_all_attribute_with_value(product):
    attr_str = ""
    for attr_name, attr_value in product.items():
        attr_str += f"attribute: {attr_name}, value: {attr_value}\n"
    return attr_str

def generate_response(
        query_node: QueryNode,
        model_path: str,
        budget: Budget,
    ) -> str:
    """
    Generate response
    """
    if query_node.response != "":
        return query_node.response

    generate_response_chain = get_model(model_path, ResponseSchema, sensitive=True, budget=budget)
    
    query = query_node.query
    query_generation_extra_info = query_node.query_generation_extra_info
    product = query_generation_extra_info["product"]
        
    # build product information
    attributes_str = get_all_attribute_with_value(product)

    # get response
    response = generate_response_chain(
        RESPONSE_GEN_PROMPT.format(attributes=attributes_str, query=query)
    )

    return response["response"]
