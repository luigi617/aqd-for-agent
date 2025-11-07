
from typing import List, Optional, Tuple
from pydantic import Field, BaseModel
from aqd.schema import PromptNode, TreeNode
from models.model import get_model


REFLECTIVE_PROMPT = """
You are an expert assistant participating in an iterative prompt optimization process for the e-commerce domain.
Your role is to refine and create a new prompt that guide the generation of user queries. These prompts are used exclusively for generating user queries, and random attribute names will be provided for this purpose.

Each generated user query is evaluated by a query filtering policy to ensure plausibility.
Queries that pass the filter receive an LLM-generated answer, which is then evaluated according to predefined *preference criteria*.

Your goal is to design a new prompt that produces user queries:
1. Likely to pass the filtering policy (i.e., not be filtered out).
2. Likely to produce unpreferred responses from the LLM based on the unpreferred criteria.

Current prompt:
{current_prompt}

Previously proposed new prompts (do NOT reuse or echo these):
{prev_prompts}

Query filter feedback (reasoning):
{filter_feedback_reasoning}

Prompt patterns contributing to query filtering:
{filter_feedback_contributing_prompt_patterns}

Improvement suggestions for avoiding filtering:
{filter_feedback_improvement_suggestions}

Unpreferred response feedback (reasoning):
{unpreferred_feedback_reasoning}

Prompt patterns contributing to unpreferred responses:
{unpreferred_feedback_contributing_prompt_patterns}

Improvement suggestions for enhancing prompts to generate user queries that are more likely to produce non-preferred responses:
{unpreferred_feedback_improvement_suggestions}


prompts (follow all):
- Avoid copying phrases from prior prompts.
- If prior prompts revealed a generalizable tactic that worked, incorporate it (in your own wording).
- Ensure the new prompt is self-contained, actionable, and unambiguous for the assistant to follow.
- Do not repeat any content listed under "Previously proposed new prompts".
- The new prompt should be generalizable and not tailored to specific examples from the feedback.

Output Format:
Return first your thought process explaining how to adjust the prompt to produce unfiltered queries that are likely to generate non-preferred answers. Then, provide a self-contained, actionable prompt designed for generating unfiltered user queries. The new prompt should be 1–3 sentences longer than the current prompts.
"""
    
class ReflectResponseSchema(BaseModel):
    reflection: str = Field(description="Thought process on how to adjust the prompt to produce unfiltered queries that are likely to generate unpreferred answers.")
    new_prompts: str = Field(
        description=(
            "Self-contained, actionable prompt text for generating unfiltered user queries"
            "The prompt should be 1–3 sentences longer than the current prompts. "
            "Write it as a clear directive in imperative form."
            "Do not include explanations, reasoning, or meta commentary."
        )
    )

def get_previous_prompts(proposed_prompts):
    prev_prompts = "\n".join([
        f"- {inst}" for inst in proposed_prompts
    ])
    if not prev_prompts:
        prev_prompts = "No prompts were generated"
    
    return prev_prompts

def refine_prompt(
    prompt_node: PromptNode,
    model_path: str, 
    n_new_prompts: int = 3
) -> List[Tuple[str, str]]:
    
    reflect_chain = get_model(model_path, ReflectResponseSchema)

    prompt = prompt_node.prompt


    filter_feedback_reason = prompt_node.query_filter_reason
    filter_feedback_contributing_prompt_patterns = "\n".join([
        f"- {pattern}" for pattern in prompt_node.query_filter_contributing_prompt_patterns
    ])
    filter_feedback_improvement_suggestions = "\n".join([
        f"- {suggestion}" for suggestion in prompt_node.query_filter_improvement_suggestions
    ])

    unpreferred_feedback_reason = prompt_node.unpreferred_response_reason
    unpreferred_feedback_contributing_prompt_patterns = "\n".join([
        f"- {pattern}" for pattern in prompt_node.unpreferred_response_contributing_prompt_patterns
    ])
    unpreferred_feedback_improvement_suggestions = "\n".join([
        f"- {suggestion}" for suggestion in prompt_node.unpreferred_response_improvement_suggestions
    ])
    

    proposed_prompts = []
    for _ in range(n_new_prompts):
        
        prev_prompts = get_previous_prompts(proposed_prompts)

        new_prompt = reflect_chain(
            REFLECTIVE_PROMPT.format(
                current_prompt = prompt, 
                filter_feedback_reasoning = filter_feedback_reason,
                filter_feedback_contributing_prompt_patterns = filter_feedback_contributing_prompt_patterns,
                filter_feedback_improvement_suggestions = filter_feedback_improvement_suggestions,
                unpreferred_feedback_reasoning = unpreferred_feedback_reason,
                unpreferred_feedback_contributing_prompt_patterns = unpreferred_feedback_contributing_prompt_patterns,
                unpreferred_feedback_improvement_suggestions = unpreferred_feedback_improvement_suggestions,
                prev_prompts = prev_prompts,
            )
        )

        proposed_prompts.append((new_prompt['new_prompts'], new_prompt["reflection"]))

    return proposed_prompts