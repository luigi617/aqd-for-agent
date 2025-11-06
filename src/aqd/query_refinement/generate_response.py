from collections import defaultdict
from tqdm import tqdm
from pipeline.misc.graph_common import *
from pipeline.misc.model import get_basic_model
from pydantic import Field, BaseModel
from pydantic import BaseModel

TEST_PROMPT = """
Provide a direct answer to the user's query based on the provided attributes.

Product attributes:
{attributes}

User query:
{query}

Instructions (follow all)
- Do NOT mention field names or say “according to the attributes”.
- Do not hallucinate any information not present.
"""


class TestResponseSchema(BaseModel):
    response: str = Field(description="A generated response to the query.")

def generate_response(
        state: GraphState, 
        model_path: str, 
        thread_n: int,
    ) -> GraphState:

    # get model (sensitive since involves full product information)
    test_chain = get_basic_model(model_path, TestResponseSchema, sensitive=True)
    prompt = state["prompt_pool"][-1]        

    state["prompts"][prompt] = state["prompts"].get(prompt, PromptState.get_default())

    for i in tqdm(range(len(state['prompts'][prompt]['filtered_queries'])), desc="Generating responses"):
        if not is_unfilled(state, 'responses', prompt, i):
            # already has response
            continue
        
        if state['prompts'][prompt]['filtered_queries'][i][0] == FILTERED:
            # filtered query, skip
            state["prompts"][prompt]["responses"].append("")
            continue

        query, _ = state['prompts'][prompt]['filtered_queries'][i]
        product = state['prompts'][prompt]['products'][i]
        

        # build product information
        attributes_str = get_attr_str(product)

        # get response
        response = test_chain(
            TEST_PROMPT.format(attributes=attributes_str, query=query)
        )['response']

        # modify state
        state["prompts"][prompt]["responses"].append(response)
    

    return state