import json
import os
from dotenv import load_dotenv
from src.aqd.prompt_search.generate_queries import generate_queries
from src.args_parser import argument_parser
from src.config import Config
from src.aqd.query_refinement.query_refinement import QueryRefinement
from src.aqd.schema import PromptNode, QueryNode

load_dotenv(override=True)

config: Config = argument_parser()


base_path = config.output_path

prompt_node_0 = "Create a user question related to online shopping that a customer might ask about a product."
prompt_node_2 = "Create one realistic shopper query about a product and its attributes. Make it a single short sentence or phrase (no backstory, no lists, no SKUs or code) that combines at least two attribute-related asks\u2014include one 'how' or step request, introduce an ambiguity or a likely-missing detail, and ask for a source or next step if the info is unavailable. Output only the single query string and nothing else."
prompt_node_3 = "Write exactly one realistic, concise shopper query (2\u201315 words, or one short sentence) a customer might type about a product and its attributes. The query must combine two attribute-related asks (for example, a technical/spec plus a purchase/action or a warranty/usage question) and include at least one ambiguous or likely-missing detail that forces inference; do not include backstory, lists, SKUs, code, or extra commentary. Output only the single query string and nothing else."
prompt_node_5 = "Write one short, natural shopper question (a single brief sentence or phrase, no backstory, no lists, SKUs, code, or technical jargon) about the product and its attributes. Include at least three attribute-related asks, one brief \u201chow\u201d that requests a numbered, step-by-step verification method with measurable checks or pass/fail thresholds, introduce one missing numeric detail or conflicting constraint that forces trade\u2011off reasoning, ask for the exact kind of authoritative source needed to confirm the claim (standard/regulation/document title or URL), and request one precise short sentence to send the seller to obtain any missing data or photos. Output only the single query string and nothing else."
prompt_node_8 = "Write exactly one realistic shopper query (6\u201325 words, or one to two short sentences) a customer might type about a product. Ask about two product-related concerns expressed naturally (for example, a spec plus a how-to or a policy plus a measurement), start with an open-ended word (what/how/which), avoid yes/no openings, include at least one numeric measure or a user constraint that may require inference, and if the info could be missing ask for alternatives or where to check; do not include backstory, lists, SKUs, code, or technical jargon. Output only the single query string and nothing else."
prompt_node_12 = "Write one short, natural shopper question (a single brief sentence or phrase; no backstory, no lists, SKUs, code, or technical jargon) about the product and its attributes. Include at most two attribute-related asks: make one a subjective user-experience concern (comfort, fit, noise, etc.) and the other a factual or appearance-related attribute, and phrase them so the main goal is mildly ambiguous or presents a simple trade-off (for example, \u201cI want comfort but care about looks\u201d). You may optionally add a casual \u201chow can I check?\u201d but do not request numbered steps, exact standards, certificate IDs, URLs, or a seller message; if asking for a source, name only a general document type (e.g., \"manufacturer spec sheet\" or \"warranty\"). Output only the single query string and nothing else."



prompt_nodes = [PromptNode(prompt_node_0), PromptNode(prompt_node_2), PromptNode(prompt_node_5)]

res = []
for prompt_node in prompt_nodes:
    n_to_be_generated_queries = 1
    
    queries = generate_queries(
        prompt_node,
        config.generate_model,
        config.budget,
        config.products,
        0,
        n_to_be_generated_queries,
    )
    
    for query, product, attribute, product_idx in queries:
        query_generation_extra_info = {"product": product, "attribute": attribute, "product_idx": product_idx}
        prompt_node.queries.append(QueryNode(query, query_generation_extra_info))

    for query_node in prompt_node.queries:
        
        query_refinement = QueryRefinement(
            query_node=query_node,
            query_tree_node=None,
            config=config
        )
        query_node_res = query_refinement.run(max_depth=config.n_query_refinement_iterations,
                                                n_workers=config.n_thread_query_refiment)
        
        prompt_node.query_refinement_map[query_node] = query_node_res
    res.append(prompt_node.to_dict())


save_path = base_path.parent / "0-2-5" / base_path.name
save_path.parent.mkdir(parents=True, exist_ok=True)
os.makedirs(save_path.parent, exist_ok=True)
with open(save_path, 'w') as f:
    json.dump(res, f, indent=4)

budget_save_path = save_path.with_name("budget.json")
with open(budget_save_path, 'w') as f:
    json.dump(config.budget.to_dict(), f, indent=4)















prompt_nodes = [PromptNode(prompt_node_0), PromptNode(prompt_node_2), PromptNode(prompt_node_12)]

res = []
for prompt_node in prompt_nodes:
    n_to_be_generated_queries = 1
    
    queries = generate_queries(
        prompt_node,
        config.generate_model,
        config.budget,
        config.products,
        0,
        n_to_be_generated_queries,
    )
    
    for query, product, attribute, product_idx in queries:
        query_generation_extra_info = {"product": product, "attribute": attribute, "product_idx": product_idx}
        prompt_node.queries.append(QueryNode(query, query_generation_extra_info))

    for query_node in prompt_node.queries:
        
        query_refinement = QueryRefinement(
            query_node=query_node,
            query_tree_node=None,
            config=config
        )
        query_node_res = query_refinement.run(max_depth=config.n_query_refinement_iterations,
                                                n_workers=config.n_thread_query_refiment)
        
        prompt_node.query_refinement_map[query_node] = query_node_res
    res.append(prompt_node.to_dict())


save_path = base_path.parent / "0-2-12" / base_path.name
save_path.parent.mkdir(parents=True, exist_ok=True)
os.makedirs(save_path.parent, exist_ok=True)
with open(save_path, 'w') as f:
    json.dump(res, f, indent=4)

budget_save_path = save_path.with_name("budget.json")
with open(budget_save_path, 'w') as f:
    json.dump(config.budget.to_dict(), f, indent=4)














prompt_nodes = [PromptNode(prompt_node_0), PromptNode(prompt_node_3), PromptNode(prompt_node_8)]

res = []
for prompt_node in prompt_nodes:
    n_to_be_generated_queries = 1
    
    queries = generate_queries(
        prompt_node,
        config.generate_model,
        config.budget,
        config.products,
        0,
        n_to_be_generated_queries,
    )
    
    for query, product, attribute, product_idx in queries:
        query_generation_extra_info = {"product": product, "attribute": attribute, "product_idx": product_idx}
        prompt_node.queries.append(QueryNode(query, query_generation_extra_info))

    for query_node in prompt_node.queries:
        
        query_refinement = QueryRefinement(
            query_node=query_node,
            query_tree_node=None,
            config=config
        )
        query_node_res = query_refinement.run(max_depth=config.n_query_refinement_iterations,
                                                n_workers=config.n_thread_query_refiment)
        
        prompt_node.query_refinement_map[query_node] = query_node_res
    res.append(prompt_node.to_dict())


save_path = base_path.parent / "0-3-8" / base_path.name
save_path.parent.mkdir(parents=True, exist_ok=True)
os.makedirs(save_path.parent, exist_ok=True)
with open(save_path, 'w') as f:
    json.dump(res, f, indent=4)

budget_save_path = save_path.with_name("budget.json")
with open(budget_save_path, 'w') as f:
    json.dump(config.budget.to_dict(), f, indent=4)


