from concurrent.futures import ThreadPoolExecutor, as_completed

from src.aqd.query_refinement.evaluate_responses import evaluate_responses
from src.aqd.query_refinement.generate_response import generate_response
from src.ace.ace import ACE
from src.aqd.prompt_search.generate_queries import generate_queries
from src.aqd.schema import PromptNode, QueryNode, TreeNode
from src.config import Config

class DataProcessor:
    
    def compute_score(self, config: Config, prompt_node: PromptNode):
        queries = generate_queries(
            prompt_node,
            config.generate_model,
            config.budget,
            config.products,
            0,
            len(config.products),
        )

        def process_one(item):
            query, product, attribute, product_idx = item

            query_generation_extra_info = {
                "product": product,
                "attribute": attribute,
                "product_idx": product_idx,
            }
            query_node = QueryNode(query, query_generation_extra_info)

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
            return query_node, query_node_score
    
        scores = []
        query_nodes = []

        # Pick a sensible default. If these calls are mostly network/IO bound, threads help a lot.
        max_workers =  min(config.n_threads, (len(queries)))

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(process_one, item) for item in queries]

            for fut in as_completed(futures):
                query_node, query_node_score = fut.result()

                # IMPORTANT: only mutate shared structures in the main thread
                prompt_node.queries.append(query_node)
                prompt_node.query_refinement_map[query_node] = TreeNode(query_node)

                query_nodes.append(query_node)
                scores.append(query_node_score)

        prompt_node.score = sum(scores) / len(scores) if scores else 0
        return prompt_node.score

def run_ace(config: Config):
    print(f"ACE: START")
    data_processor = DataProcessor()
    ace = ACE(config=config)
    tree_prompt_node = TreeNode(None)
    test_samples = [tree_prompt_node]
    
    ace_res = ace.run(mode="online", test_samples=test_samples, data_processor=data_processor)
    print(f"ACE: DONE")
    return ace_res