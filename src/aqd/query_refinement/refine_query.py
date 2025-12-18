from typing import List, Tuple
from tqdm import tqdm
from pydantic import Field, BaseModel
from pydantic import BaseModel
from src.aqd.budget import Budget
from src.models.model import get_model
from src.aqd.schema import TreeNode

# Stretegy instructions taken from https://arxiv.org/pdf/2310.13345

QUERY_PERTURBATION_PROMPT = """
Given the current query, generate a new query by the guidance of the perturbation strategy while keeping the meaning of the original query.

Current query:
{current_query}

Perturbation strategy:
{strategy}

Output Format:
Return the new query.
"""

QUERY_PROFILE_PROMPT = """You are rewriting a user query to reflect a specific customer persona. 
Original query: 
{current_query}

Customer persona:
{profile}

Rules:
Rewrite the original query to reflect this customer persona, including any motivations or considerations that fit naturally. The rewritten question should:
- Do not include any identity information of the customer.
- Do not repeat the exact instruction in the persona.
- Try to personalize the queestion to the persona's background, such as job information, household, event or activities you did or going to do.
- Keep the original factual intent and product focus.
- Embed personal, emotional, or contextual details drawn from the persona.  
- Maintain a natural tone and fluency.
- Be locally refined, not entirely reformulated.
- Be concise (1 - 2 sentences).

"""

QUERY_SCENARIO_PROMPT = """You are rewriting a user query to reflect a specific scenario. 
Original query: 
{current_query}

Scenario:
{scenario}

Rules:
Rewrite the original query in the specified scenario. The rewritten question should:
- Apply the scenario in relevant way to the original query. If the scenario is not directly related to the original query, make a connection between the scenario and the original query.
- Keep the original factual intent and product focus.
- Be locally refined, not entirely reformulated.
"""

QUERY_TONE_PROMPT = """You are rewriting a user query to reflect a specific tone. 
Original query: 
{current_query}

Tone:
{tone}

Rules:
Rewrite the original query in the specified tone. The rewritten question should:
- Keep the original factual intent and product focus.
- Be locally refined, not entirely reformulated.
"""


class QueryRefinementSchema(BaseModel):
    query: str = Field(description="Refined query.")


def refine_query(
        tree_node: TreeNode,
        model_path: str,
        budget: Budget,
        query_refinement_strategies: list,
        strategy_prompt_map: dict,
    ) -> List[Tuple[str, str]]:

    refine_query_chain = get_model(model_path, QueryRefinementSchema, budget=budget)
    
    query_node = tree_node.value
    query = query_node.query

    proposed_queries = []
    for strategy in query_refinement_strategies:
        strategy_prompt = None
        if strategy_prompt_map[strategy]["prompt_name"] == "perturbation":
            strategy_prompt = QUERY_PERTURBATION_PROMPT.format(
                current_query=query,
                strategy=strategy,
            )
        elif strategy_prompt_map[strategy]["prompt_name"] == "profile":
            strategy_prompt = QUERY_PROFILE_PROMPT.format(
                current_query=query,
                profile=strategy,
            )
        elif strategy_prompt_map[strategy]["prompt_name"] == "scenario":
            strategy_prompt = QUERY_SCENARIO_PROMPT.format(
                current_query=query,
                scenario=strategy,
            )
        elif strategy_prompt_map[strategy]["prompt_name"] == "tone":
            strategy_prompt = QUERY_TONE_PROMPT.format(
                current_query=query,
                tone=strategy,
            )


        if not strategy_prompt:
            raise Exception("strategy not recognized")
        
        refinement = refine_query_chain(
            strategy_prompt,
            budget_extra_info={"desc": f"Refine query ({query}) for strategy: {strategy}"}
        )
        proposed_queries.append((refinement["query"], strategy))


    return proposed_queries
