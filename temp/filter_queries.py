from collections import defaultdict

from tqdm import tqdm
from pipeline.misc.graph_common import (
    GraphState,
    PromptState,
    is_unfilled,
    FILTERED
)
from pipeline.misc.model import get_basic_model
from pydantic import Field, BaseModel
from pydantic import BaseModel


FILTER_QUERY_PROMPT = """\
You are a filter that decides whether the user's query satisfies the filtering definition.

Query:
{query}

Filtering definition:
{filtering_policy}
"""

class FilteredQuerySchema(BaseModel):
    reasoning: str = Field(description="reasoning behind the query satisfing or not satisfying the filtering definition.")
    satisfy_filter: bool = Field(description="True if the query satisfies the filtering definition, False otherwise.")

def filter_queries(
        state: GraphState, 
        model_path: str, 
        thread_n: int,
        filtering_policy: str,
    ) -> GraphState:
    """
    Filters queries based on realism definition.
    Returns updates to state with new filtered queries.
    """
    query_chain = get_basic_model(model_path, FilteredQuerySchema)
    # get current queries responsible by thread
    prompt = state["prompt_pool"][-1]
    
    state["prompts"][prompt] = state["prompts"].get(prompt, PromptState.get_default())

    # filter queries and update state
    for i in tqdm(range(len(state['prompts'][prompt]['template_queries'])), desc="Filtering queries"):
        if not is_unfilled(state, 'filtered_queries', prompt, i):
            # already filtered
            continue
        query = state['prompts'][prompt]['template_queries'][i]

        decision = query_chain(
            FILTER_QUERY_PROMPT.format(
                query=query,
                filtering_policy=filtering_policy
            )
        )
        filtered_query = query if decision['satisfy_filter'] else FILTERED
        state["prompts"][prompt]['filtered_queries'].append((filtered_query, decision['reasoning']))
    
    return state