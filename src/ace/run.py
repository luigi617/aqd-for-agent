from concurrent.futures import ThreadPoolExecutor, as_completed

from src.aqd.query_refinement.evaluate_responses import evaluate_responses
from src.aqd.query_refinement.generate_response import generate_response
from src.ace.ace import ACE
from src.aqd.prompt_search.generate_queries import generate_queries
from src.aqd.schema import PromptNode, QueryNode, TreeNode
from src.config import Config

class DataProcessor:
    
    def compute_score(self, config: Config, prompt_node: PromptNode):
        print(f"Start computing score for prompt node...")
        query_dict = {query.query_generation_extra_info["product_idx"]: query for query in prompt_node.queries}

        def process_one(item):
            prompt_node, product_idx = item
            print(f"Processing one query product idx: {product_idx}...")
            if product_idx not in query_dict:
                queries = generate_queries(
                    prompt_node,
                    config.generate_model,
                    config.budget,
                    config.products,
                    product_idx,
                    1,
                )
                query, product, attribute, _product_idx = queries[0]
                query_generation_extra_info = {
                    "product": product,
                    "attribute": attribute,
                    "product_idx": product_idx,
                }
                query_node = QueryNode(query, query_generation_extra_info)
            else:
                query_node = query_dict[product_idx]

            response = generate_response(query_node, config.target_model, config.budget)
            query_node.response = response
            criteria_chosen, reason, individual_evals = evaluate_responses(
                query_node,
                config.evaluation_models,
                config.refinement_model,
                config.budget,
                config.preferred_definition,
                config.unpreferred_definition,
            )

            query_node.set_is_unpreferred(criteria_chosen, config.unpreferred_definition)
            query_node.evaluation = criteria_chosen
            query_node.evaluation_reason = reason
            query_node.individual_evaluations = individual_evals
            query_node_score = query_node.get_score(config.preference_score_map)
            print(f"Finished processing one query item: {item}.")
            return query_node, query_node_score
    
        scores = []

        # Pick a sensible default. If these calls are mostly network/IO bound, threads help a lot.
        max_workers =  min(config.n_threads, (len(config.products)))

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for product_idx in range(len(config.products)):
                futures.append(ex.submit(process_one, (prompt_node, product_idx)))

            for fut in as_completed(futures):
                query_node, query_node_score = fut.result()
                product_idx = query_node.query_generation_extra_info["product_idx"]
                for i in range(len(prompt_node.queries)):
                    if prompt_node.queries[i].query_generation_extra_info["product_idx"] == product_idx:
                        q = prompt_node.queries.pop(i)
                        prompt_node.query_refinement_map.pop(q)
                        break
                # IMPORTANT: only mutate shared structures in the main thread
                prompt_node.queries.append(query_node)
                prompt_node.query_refinement_map[query_node] = TreeNode(query_node)

                scores.append(query_node_score)

        prompt_node.score = sum(scores) / len(scores) if scores else 0
        return prompt_node.score

def run_ace(config: Config):
    print(f"ACE: START")
    data_processor = DataProcessor()
    ace = ACE(config=config)
    if config.prev_result:
        tree_prompt_node = ace.load_prev_result(config.prev_result)
    else:
        tree_prompt_node = TreeNode(None)
    test_samples = [tree_prompt_node]
    
    ace_res = ace.run(mode="online", test_samples=test_samples, data_processor=data_processor)
    print(f"ACE: DONE")
    return ace_res