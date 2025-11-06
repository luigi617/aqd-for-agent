from collections import defaultdict
from tqdm import tqdm
from pipeline.misc.graph_common import (
    GraphState,
    PromptState,
    FILTERED
)
from pipeline.misc.model import get_basic_model
from typing import List, Optional
from pydantic import Field, BaseModel

# https://arxiv.org/pdf/2507.19457 


QUERY_FILTER_FEEDBACK_PROMPT = """
You are an expert assistant participating in an iterative instruction optimization process within the e-commerce domain.
Your role is to refine and create instructions that guide the generation of user queries. These instructions are used exclusively for generating user queries, and random attribute names will be provided for this purpose.

Some of the generated user queries have been filtered out based on a defined query filtering policy.
Your task is to analyze and explain why the current instruction might have led to those filtered-out queries, and to propose concrete improvements.

Current instruction set:
{current_instructions}

Filtered out queries:
{filtered_out_queries}

Query filtering policy:
{filtering_policy}
"""

UNPREFERRED_RESPONSE_FEEDBACK_PROMPT = """
You are an expert assistant in an iterative instruction-optimization process for the e-commerce domain.
Your role is to refine and create instructions that guide the generation of user queries. These instructions are used exclusively for generating user queries, and random attribute names may be provided as inputs.

Some generated user queries were answered by a large language model in a non-preferred way, as defined below.
Your task is to analyze and explain why the current instruction likely caused user queries that resulted in non-preferred responses. Additionally, propose concrete improvements to the instruction to ensure that produce user queries are more likely to elicit preferred responses.

Current instruction set:
{current_instructions}

User queries with non-preferred responses and reasons:
{unpreferred_responses}

Non-preferred criteria:
{unpreferred_criteria}
"""



REFLECTIVE_PROMPT = """
You are an expert assistant participating in an iterative instruction optimization process for the e-commerce domain.
Your role is to refine and create a new instructions that guide the generation of user queries. These instructions are used exclusively for generating user queries, and random attribute names will be provided for this purpose.

Each generated user query is evaluated by a query filtering policy to ensure plausibility.
Queries that pass the filter receive an LLM-generated answer, which is then evaluated according to predefined *preference criteria*.

Your goal is to design a new instruction that produces user queries:
1. Likely to pass the filtering policy (i.e., not be filtered out).
2. Likely to produce unpreferred responses from the LLM based on the unpreferred criteria.

Current instruction:
{current_instructions}

Previously proposed new instructions (do NOT reuse or echo these):
{prev_instructions}

Query filter feedback (reasoning):
{filter_feedback_reasoning}

Instruction patterns contributing to query filtering:
{filter_feedback_contributing_instruction_patterns}

Improvement suggestions for avoiding filtering:
{filter_feedback_improvement_suggestions}

Unpreferred response feedback (reasoning):
{unpreferred_feedback_reasoning}

Instruction patterns contributing to unpreferred responses:
{unpreferred_feedback_contributing_instruction_patterns}

Improvement suggestions for enhancing instructions to generate user queries that are more likely to produce non-preferred responses:
{unpreferred_feedback_improvement_suggestions}


Instructions (follow all):
- Avoid copying phrases from prior instructions.
- If prior instructions revealed a generalizable tactic that worked, incorporate it (in your own wording).
- Ensure the new instruction is self-contained, actionable, and unambiguous for the assistant to follow.
- Do not repeat any content listed under "Previously proposed new instructions".
- The new instruction should be generalizable and not tailored to specific examples from the feedback.
"""




class QueryFilterFeedbackSchema(BaseModel):
    
    reasoning: str = Field(description="Explanation of why the current instruction produced queries not satisfying query filtering policy.")
    
    contributing_instruction_patterns: List[str] = Field(
        default_factory=list,
        description="Specific patterns, phrases, or logic in the instruction that likely caused queries to be filtered out."
    )
    
    improvement_suggestions: Optional[List[str]] = Field(
        default=None,
        description="Optional guidance on how to refine or reword the instruction to prevent queries to be filtered out."
    )

class QueryUnpreferredFeedbackSchema(BaseModel):
    
    reasoning: str = Field(description="Explanation of how the current instruction produces queries that are answered in a non-preferred way.")
    contributing_instruction_patterns: List[str] = Field(
        default_factory=list,
        description="Phrases/rules in the instruction that generate queries that promote non-preferred responses."
    )
    improvement_suggestions: Optional[List[str]] = Field(
        default=None,
        description="Optional guidance on how to refine the instruction to generate user queries that are more likely to produce unpreferred response."
    )

    
class ReflectResponseSchema(BaseModel):
    reflection: str = Field(description="Thought process on how to adjust the instruction to produce unfiltered queries that are likely to generate unpreferred answers.")
    difference_from_current: str = Field(description="Concise note on what changed relative to current instructions")
    new_instructions: str = Field(
        description=(
            "Self-contained, actionable instruction text for generating unfiltered user queries"
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

def get_filtered_out_queries(template_queries, queries):
    filtered_out_queries = ""
    for original_query, (question, reason) in zip(template_queries, queries):
        if question == FILTERED:
            filtered_out_queries += (
                f"Query: {original_query}\n"
                f"Reason for violating query filtering policy: {reason}\n\n"
            )

    if not filtered_out_queries:
        filtered_out_queries = "No filtered-out queries generated"
    return filtered_out_queries

def reflective_mutation(
    state: GraphState, 
    model_path: str, 
    preferred_definition: dict,
    unpreferred_definition: dict,
    filtering_policy: str,
    thread_n: int,
    n_new_prompts: int = 3
) -> GraphState:
    
    query_filtering_chain = get_basic_model(model_path, QueryFilterFeedbackSchema)
    unpreferred_response_chain = get_basic_model(model_path, QueryUnpreferredFeedbackSchema)
    reflect_chain = get_basic_model(model_path, ReflectResponseSchema)

    prompt = state["prompt_pool"][-1]
    state["prompts"][prompt] = state["prompts"].get(prompt, PromptState.get_default())

   

    template_queries = state['prompts'][prompt]['template_queries']
    queries = state['prompts'][prompt]['filtered_queries']
    responses = state['prompts'][prompt]['responses']
    response_evals = state['prompts'][prompt]['response_evals']

    unpreferred_criteria = []
    for k, v in unpreferred_definition.items():
        unpreferred_criteria.extend(v)
    unpreferred_criteria = "\n".join([f"- {c}" for c in unpreferred_criteria])
    
    history = []

    # get feedback: unpreferred and query filter

    query_filtering_feedback = query_filtering_chain(
        QUERY_FILTER_FEEDBACK_PROMPT.format(
            current_instructions=prompt, 
            filtered_out_queries=get_filtered_out_queries(template_queries, queries),
            filtering_policy=filtering_policy
        )
    )

    unpreferred_responses = get_unpreferred_str(queries, responses, response_evals, unpreferred_definition)
    unpreferred_response_feedback = unpreferred_response_chain(
        UNPREFERRED_RESPONSE_FEEDBACK_PROMPT.format(
            current_instructions=prompt, 
            unpreferred_responses=unpreferred_responses,
            unpreferred_criteria=unpreferred_criteria
        )
    )
    with tqdm(total=1, desc="Getting query filtering feedback") as pbar:
        state["prompts"][prompt]["query_filtering_feedback"] = (
            query_filtering_feedback["reasoning"],
            query_filtering_feedback["contributing_instruction_patterns"],
            query_filtering_feedback.get("improvement_suggestions", None)
        )
        pbar.update(1)

    with tqdm(total=1, desc="Getting unpreferred response feedback") as pbar:
        state["prompts"][prompt]["unpreferred_response_feedback"] = (
            unpreferred_response_feedback["reasoning"],
            unpreferred_response_feedback["contributing_instruction_patterns"],
            unpreferred_response_feedback.get("improvement_suggestions", None)
        )
        pbar.update(1)

    for _ in tqdm(range(n_new_prompts), desc="Generating reflective mutations"):


        # get feedback string and new instructions
        filter_feedback_contributing_instruction_patterns = "\n".join([
            f"- {pattern}" for pattern in query_filtering_feedback["contributing_instruction_patterns"]
        ])
        filter_feedback_improvement_suggestions = "\n".join([
            f"- {suggestion}" for suggestion in (query_filtering_feedback.get("improvement_suggestions", ["No suggestions provided"]))
        ])
        unpreferred_feedback_contributing_instruction_patterns = "\n".join([
            f"- {pattern}" for pattern in unpreferred_response_feedback["contributing_instruction_patterns"]
        ])
        unpreferred_feedback_improvement_suggestions = "\n".join([
            f"- {suggestion}" for suggestion in (unpreferred_response_feedback.get("improvement_suggestions", ["No suggestions provided"]))
        ])
        prev_instructions = "\n".join([
            f"- {inst}" for inst in history
        ])
        new_instruction = reflect_chain(
            REFLECTIVE_PROMPT.format(
                current_instructions = prompt, 
                filter_feedback_reasoning = query_filtering_feedback["reasoning"],
                filter_feedback_contributing_instruction_patterns = filter_feedback_contributing_instruction_patterns,
                filter_feedback_improvement_suggestions = filter_feedback_improvement_suggestions,
                unpreferred_feedback_reasoning = unpreferred_response_feedback["reasoning"],
                unpreferred_feedback_contributing_instruction_patterns = unpreferred_feedback_contributing_instruction_patterns,
                unpreferred_feedback_improvement_suggestions = unpreferred_feedback_improvement_suggestions,
                prev_instructions = prev_instructions,
            )
        )
        history.append(new_instruction['new_instructions'])

        # add new instructions to state
        state["prompts"][prompt]["reflections"].append(
            (new_instruction['reflection'], new_instruction['new_instructions'])
        )


    return state