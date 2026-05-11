from pydantic import Field, BaseModel
from pydantic import BaseModel
from src.aqd.budget import Budget
from src.aqd.schema import QueryNode
from src.models.model import get_model

RESPONSE_GEN_PROMPT = """
Provide a direct answer to the user's query based on the provided attributes.
Instructions (follow all)
- Do NOT mention field names or say “according to the attributes”.
- Do not hallucinate any information not present.

Output Format:
{{
    "response": "<The generated response to the query.>"
}}

Product attributes:
{attributes}

User query:
{query}

Return:
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
        generate_response_chain
    ) -> str:
    """
    Generate response
    """
    if query_node.response is not None:
        return query_node.response

    # generate_response_chain = get_model(model_path, ResponseSchema, sensitive=True, budget=budget)
    
    query = query_node.query
    query_generation_extra_info = query_node.query_generation_extra_info
    product = query_generation_extra_info["product"]
        
    # build product information
    attributes_str = get_all_attribute_with_value(product)
    # ensure the attributes string is not too long
    print("attributes_str length: ", len(attributes_str))
    # if len(attributes_str) > 1000:
    #     attributes_str = attributes_str[:1000]
    print("prompt length: ", len(RESPONSE_GEN_PROMPT.format(attributes=attributes_str, query=query)))

    # get response
    try:
        response = generate_response_chain(
            RESPONSE_GEN_PROMPT.format(attributes=attributes_str, query=query),
            budget_extra_info={"desc": f"Generating response for query: {query}"}
        )
        return response["response"]
    except Exception as e:
        print(f"Error generating response: {e}")
        generate_response_chain = get_model(model_path, None, sensitive=True, budget=budget)
        response = generate_response_chain(
            RESPONSE_GEN_PROMPT.format(attributes=attributes_str, query=query),
            budget_extra_info={"desc": f"Generating response for query: {query}"}
        )
        return response

