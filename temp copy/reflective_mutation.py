from collections import defaultdict
from tqdm import tqdm
from pipeline.misc.graph_common import (
    GraphState,
    PromptState,
    FILTERED
)
from pipeline.misc.model import get_basic_model
from pydantic import Field, BaseModel
from random import choice
from pydantic import BaseModel, Field
from functools import partial

# https://arxiv.org/pdf/2507.19457 
REFLECTIVE_PROMPT = """
You are an expert assistant participating in an iterative instruction optimization pipeline focused on the e-commerce domain.
Your role is to refine and create instructions that guide the generation of user queries. These instructions are used solely for user query generation, and randomly chosen attribute names will be provided for this purpose.

User query will be filtered out based on a realism policy to ensure they are plausible.
Eacy Realistic queries receives an answer from a large language model and it will be evaluated based on preference definition.

Your goal is to design a new instruction that leads to the generation of realistic user queries more likely to produce unpreferred answers from the LLM (according to the unpreferred criteria).

Current instruction set:
{current_instructions}

Feedback from previous iteration (use these to generate improved instruction):
{feedback}

Previously proposed new instructions (do NOT reuse or echo these):
{prev_instructions}

Instructions (follow all):
- Avoid copying phrases from prior instructions.
- If prior instructions revealed a generalizable tactic that worked, incorporate it (in your own wording).
- Ensure the new instruction is self-contained, actionable, and unambiguous for the assistant to follow.
- Do not repeat any content listed under "Previously proposed new instructions".
- The new instruction should be generalizable and not tailored to specific examples from the feedback.
"""

FEEDBACK_TEMPLATE = """
The below responses are good unpreferred responses with reasons why they are unpreferred:
{unpreferred_responses}
Here are the unpreferred criteria that you should aim to optimize for in the next iteration.
{unpreferred_criteria}

At the same time, ensure that realism is still present within queries.
Here were examples of queries that were filtered out due to being too unrealistic:
{unrealistic_responses}
Additionally, here is the definition used to determine realism:
{filtering_policy}
"""


class ReflectResponseSchema(BaseModel):
    reflection: str = Field(description="Thought process on how to adjust the instruction to produce realistic queries that are likely to generate unpreferred answers.")
    difference_from_current: str = Field(description="Concise note on what changed relative to current instructions")
    new_instructions: str = Field(
        description=(
            "Self-contained, actionable instruction text for generating realistic user queries"
            "The instruction should be 1–3 sentences longer than the current instructions. "
            "Write it as a clear directive in imperative form."
            "Do not include explanations, reasoning, or meta commentary."
        )
    )

def get_unpreferred_str(queries, responses, response_evals, unpreferred_definition):
    unpreferred_builder = ""
    for ((question, _), response, response_eval) in zip(queries, responses, response_evals):
        if question != FILTERED:
            # have to unpack inside in case evalu is None due to filtered question
            (criteria_chosen, reason) = response_eval
            is_unpreferred = False
            for category, _ in criteria_chosen:
                if category in unpreferred_definition:
                    is_unpreferred = True

            if is_unpreferred:
                unpreferred_builder += (
                    f"Query: {question}\n"
                    f"Response: {response}\n"
                    f"Reason for unpreferred: {reason}\n\n"
                )

    if not unpreferred_builder:
        unpreferred_builder = "No unpreferred responses were generated"
    return unpreferred_builder

def get_unrealistic_str(template_queries, queries):
    unrealistic_builder = ""
    for original_query, (question, reason) in zip(template_queries, queries):
        if question == FILTERED:
            unrealistic_builder += (
                f"Query: {original_query}\n"
                f"Reason for lack of realism: {reason}\n\n"
            )

    if not unrealistic_builder:
        unrealistic_builder = "No unrealistic queries generated"
    return unrealistic_builder

def reflective_mutation(
    state: GraphState, 
    model_path: str, 
    preferred_definition: dict,
    unpreferred_definition: dict,
    filtering_policy: str,
    thread_n: int,
    n_new_prompts: int = 3
) -> GraphState:
    
    test_chain = get_basic_model(model_path, ReflectResponseSchema)

    prompt = state["prompt_pool"][-1]
    state["prompts"][prompt] = state["prompts"].get(prompt, PromptState.get_default())

   

    template_queries = state['prompts'][prompt]['template_queries']
    queries = state['prompts'][prompt]['filtered_queries']
    responses = state['prompts'][prompt]['responses']
    response_evals = state['prompts'][prompt]['response_evals']


    history = []

    for _ in tqdm(range(n_new_prompts), desc="Generating reflective mutations"):
        
        # get feedback (unpreferred, unrealistic)
        
        unpreferred_builder = get_unpreferred_str(queries, responses, response_evals, unpreferred_definition)

        unrealistic_builder = get_unrealistic_str(template_queries, queries)
        

        # select random preferred criteria for optimization
        unpreferred_criteria = []
        for k, v in unpreferred_definition.items():
            unpreferred_criteria.extend(v)
        unpreferred_criteria = "\n".join([f"- {c}" for c in unpreferred_criteria])

        # get feedback string and new instructions
        feedback = FEEDBACK_TEMPLATE.format(
            unpreferred_responses=unpreferred_builder, 
            unpreferred_criteria=unpreferred_criteria,
            unrealistic_responses=unrealistic_builder,
            filtering_policy=filtering_policy
        )
        new = test_chain(
            REFLECTIVE_PROMPT.format(
                current_instructions=prompt, 
                feedback=feedback, 
                prev_instructions="\n".join(history)
            )
        )
        history.append(new['new_instructions'])

        # add new instructions to state
        state["prompts"][prompt]["reflections"].append(
            (new['reflection'], new['new_instructions'])
        )


    return state