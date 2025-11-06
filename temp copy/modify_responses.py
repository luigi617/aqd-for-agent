from collections import defaultdict

from tqdm import tqdm
from pipeline.misc.graph_common import *
from pipeline.misc.model import get_basic_model
from pydantic import Field, BaseModel
from random import choice

from pipeline.nodes.evaluate_responses import EVAL_PROMPT, get_schema


CHANGE_UNPREFERRED_PROMPT = """\
Given the preferred criteria, violated criteria, and the rationale for the violation, your task is to revise the unpreferred response so that it becomes a preferred response. The revised response should answer the user’s query accurately and appropriately, based on the product’s attributes.

Query:
{query}

Unpreferred response to modify:
{unpreferred_response}

Product attributes (the only facts you may rely on):
{attr_list}

Preferred criteria to satisfy (ALL)
{preferred_criteria}

Violation(s) to fix:
{violated_criteria}

Why it was wrong:
{reason}

Instructions (follow all):
- Rewrite must be substantially different in structure and wording from the current unpreferred answer.
- You must fix EVERY item in the violation and satisfy ALL items in the preferred criteria.
- If required information is missing, state the limitation briefly and offer precise next steps or alternatives.
- Do not hallucinate.
"""


CHANGE_PREFERRED_PROMPT = """\
Given the criterion describing what makes a response unpreferred, your task is to modify the preferred response to create an unpreferred response that answer the query but clearly violates this criterion.

Query:
{query}

Preferred response to modify:
{preferred_response}

Product attributes (may be omitted/misused only as needed):
{attr_list}

Unpreferred criteria to VIOLATE (VIOLATE ALL):
{unpreferred_criteria}

Instructions (follow all):
- Rewrite must be substantially different in structure and wording from the current preferred answer.
- Do not reference or mention the original preferred response in your output.
- Use a natural tone; avoid obvious sarcasm or deliberate nonsense.
- If product attributes are provided, misuse, omit, or contradict them only as needed to satisfy the violation; otherwise keep them plausible.
- Do not hallucinate.
- Produce a single rewritten answer that violates the unpreferred criteria. If multiple criteria are supplied, violate ALL of them.
"""


class UnpreferredResponseModification(BaseModel):
    chain_of_thought: str = Field(description="Think step by step to fix all violations to produce a new response that satisfy all preferred criteria.")
    preferred_response: str = Field(description="Preferred response.")

class PreferredResponseModification(BaseModel):
    chain_of_thought: str = Field(description="Think step by step to modify given preferred response to  produce a new response that violates unpreferred criteria.")
    unpreferred_response: str = Field(description="Unpreferred response.")

def modify_responses(
    state: GraphState, 
    model_path: str, 
    eval_model_path: str, 
    thread_n: int,
    preferred_definition: dict,
    unpreferred_definition: dict,
) -> GraphState:

    # sensitive since uses full product information for rewriting prompts
    preferred_modifier_chain = get_basic_model(model_path, PreferredResponseModification, sensitive=True, retries=-1)
    unpreferred_modifier_chain = get_basic_model(model_path, UnpreferredResponseModification, sensitive=True, retries=-1)

    preference_criteria_list = []
    for criteria in preferred_definition.values():
        preference_criteria_list.extend(criteria)
    for criteria in unpreferred_definition.values():
        preference_criteria_list.extend(criteria)
    
    test_chain = get_basic_model(eval_model_path, get_schema(preference_criteria_list))
    prompt = state["prompt_pool"][-1]
    state["prompts"][prompt] = state["prompts"].get(prompt, PromptState.get_default())

    # converts categories into list to sample from in case of turning preferred to unpreferred
    preferred_categories = [k for k in preferred_definition.keys()]
    unpreferred_categories = [k for k in unpreferred_definition.keys()]

    preferred_criteria_list = []
    for criteria in preferred_definition.values():
        preferred_criteria_list.extend(criteria)


    for i in tqdm(range(len(state['prompts'][prompt]['response_evals'])), desc="Modifying responses"):
        if state["prompts"][prompt]["filtered_queries"][i][0] == FILTERED:
            state["prompts"][prompt]["modified_responses"].append("")
            state["prompts"][prompt]["modified_response_evals"].append(([], "Query was filtered"))
            continue
        if not is_unfilled(state, 'modified_responses', prompt, i):
            # already has modified response
            continue
        query = state['prompts'][prompt]['template_queries'][i]
        response = state['prompts'][prompt]['responses'][i]
        product = state['prompts'][prompt]['products'][i]
        violated_criteria, reason = state['prompts'][prompt]['response_evals'][i]
        # change unpreferred to preferred
        has_violated_criteria = any(criteria[0] in unpreferred_categories for criteria in violated_criteria)
        if has_violated_criteria:
            
            preferred_response = unpreferred_modifier_chain(
                CHANGE_UNPREFERRED_PROMPT.format(
                    query=query,
                    unpreferred_response=response,
                    attr_list=get_attr_str(product),
                    preferred_criteria="; ".join(preferred_criteria_list),
                    violated_criteria="; ".join([c[1] for c in violated_criteria if c[0] in unpreferred_categories]),
                    reason=reason,
                )
            )
            preferred = preferred_response["preferred_response"]
            chain_of_thought = preferred_response["chain_of_thought"]

            preferred_eval = test_chain(
                EVAL_PROMPT.format(
                    query=query,
                    response=preferred,
                    criteria=preference_criteria_list,
                )
            )
      
            preferred_response_categories = []
            for criterion in preferred_eval['criteria']:
                for category, criteria in {**preferred_definition, **unpreferred_definition}.items():
                    if criterion.value in criteria:
                        preferred_response_categories.append((category, criterion.value))

            # format of state data is (score, preferred, unpreferred)
            state["prompts"][prompt]["modified_responses"].append(preferred)
            state["prompts"][prompt]["modified_response_evals"].append(
                (preferred_response_categories, preferred_eval['reasoning'])
            )
        # change preferred to unpreferred
        else:
            # get new response
            category = choice(unpreferred_categories)
            unpreferred_criteria = choice(unpreferred_definition[category])
            # TODO: give entire list of unpreferred criteria to violate?
            
            unpreferred_response = preferred_modifier_chain(
                CHANGE_PREFERRED_PROMPT.format(
                    query=query,
                    preferred_response=response,
                    attr_list=get_attr_str(product),
                    unpreferred_criteria=unpreferred_criteria
                )
            )
            unpreferred = unpreferred_response["unpreferred_response"]
            chain_of_thought = unpreferred_response["chain_of_thought"]

            unpreferred_eval = test_chain(
                EVAL_PROMPT.format(
                    query=query,
                    response=unpreferred,
                    criteria=preference_criteria_list,
                )
            )
            

            unpreferred_response_categories = []
            for criterion in unpreferred_eval['criteria']:
                for category, criteria in {**preferred_definition, **unpreferred_definition}.items():
                    if criterion.value in criteria:
                        unpreferred_response_categories.append((category, criterion.value))
            
            # format of state data is (category, preferred, unpreferred)
            state["prompts"][prompt]["modified_responses"].append(unpreferred)
            state["prompts"][prompt]["modified_response_evals"].append( 
                (unpreferred_response_categories, unpreferred_eval['reasoning'])
            )

    return state