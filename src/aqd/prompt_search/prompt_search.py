import json
import os
import random
import contextvars
from typing import List, Optional
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.aqd.schema import PromptNode, QueryNode, TreeNode
from src.aqd.prompt_search.generate_queries import generate_queries
from src.aqd.prompt_search.refine_prompt import refine_prompt
from src.aqd.prompt_search.query_filter_feedback import query_filter_feedback
from src.aqd.prompt_search.unpreferred_response_feedback import unpreferred_response_feedback
from src.aqd.query_refinement.query_refinement import QueryRefinement


class PromptSearch:
    def __init__(self, prompt,
                 products,
                 n_queries,
                 generate_model,
                 refinement_model,
                 target_model,
                 filtering_model,
                 evaluation_models,
                 preferred_definition,
                 unpreferred_definition,
                 preference_score_map,
                 filtering_policy,
                 query_refinement_strategies,

                 n_prompt_refinemnet_children,
                 n_prompt_refinemnet_beam_search_size,
                 n_query_refinemnet_iterations,
                 n_query_refiment_beam_search_size,
                 n_thread_queries,
                 n_thread_query_refiment,

                 base_depth_output_path,
                 budget,
                 ):
        self.root: TreeNode = TreeNode(PromptNode(prompt))
        self.products: List[dict] = products
        self.n_queries: int = n_queries

        self.generate_model = generate_model
        self.refinement_model = refinement_model
        self.target_model = target_model
        self.filtering_model = filtering_model
        self.evaluation_models = evaluation_models

        self.preferred_definition = preferred_definition
        self.unpreferred_definition = unpreferred_definition
        self.preference_score_map = preference_score_map
        self.filtering_policy = filtering_policy
        self.query_refinement_strategies = query_refinement_strategies

        self.n_prompt_refinemnet_children = n_prompt_refinemnet_children
        self.n_prompt_refinemnet_beam_search_size = n_prompt_refinemnet_beam_search_size
        self.n_query_refinemnet_iterations = n_query_refinemnet_iterations
        self.n_query_refiment_beam_search_size = n_query_refiment_beam_search_size
        self.n_thread_queries = n_thread_queries
        self.n_thread_query_refiment = n_thread_query_refiment

        self.base_depth_output_path = base_depth_output_path
        self.budget = budget



    def _run_single_query_refinement(self, query_node, query_tree_node):
        query_refinement = QueryRefinement(
            query_node=query_node,
            query_tree_node=query_tree_node,
            target_model=self.target_model,
            filtering_model=self.filtering_model,
            refinement_model=self.refinement_model,
            evaluation_models=self.evaluation_models,
            preferred_definition=self.preferred_definition,
            unpreferred_definition=self.unpreferred_definition,
            preference_score_map=self.preference_score_map,
            filtering_policy=self.filtering_policy,
            query_refinement_strategies=self.query_refinement_strategies,
            n_query_refiment_beam_search_size=self.n_query_refiment_beam_search_size,
            budget=self.budget,
        )
        return query_node, query_refinement.run(max_depth=self.n_query_refinemnet_iterations,
                                                n_workers=self.n_thread_query_refiment)

    def _process_node(self, tree_node):
        prompt_node: PromptNode = tree_node.value
        
        with tqdm(total=2, desc="Processing prompt refinement node", unit="step") as pbar:
            prompt_node = self.generate_queries(prompt_node)
            pbar.update(1)
            max_workers = min(self.n_thread_queries, len(prompt_node.queries))
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as inner_pool:
                futures = [
                    inner_pool.submit(self._run_single_query_refinement, q, prompt_node.query_refinement_map.get(q))
                    for q in prompt_node.queries
                ]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Refining query"):
                    query_node, query_tree_node = fut.result()
                    results.append((query_node, query_tree_node))
            pbar.update(1)
            for query_node, query_tree_node in results:
                prompt_node.query_refinement_map[query_node] = query_tree_node
        
            prompt_node.get_score(self.preference_score_map)
        print("Prompt node budget: Spent:", self.budget.spent_usd, "remaining:", self.budget.remaining_usd)
        return tree_node
    
    def run(self, max_depth = 3, n_workers = 5):

        
        current_level = [self.root]
        depth = 0

        with ThreadPoolExecutor(max_workers = n_workers) as pool:
            
            while current_level:
                allow_refine = depth < max_depth
                futures = [pool.submit(self._process_node, node) for node in current_level]

                next_level = []
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Prompt search: depth {depth}"):
                    fut.result()
                
                if allow_refine:
                    prompt_tree_node_to_be_rifined = current_level if depth == 0 else self.prune_prompt_nodes(current_level)
                    for ptn in prompt_tree_node_to_be_rifined:
                        prompt_node = ptn.value
                        prompt_node = self.get_query_filter_feedback(prompt_node)
                        prompt_node = self.get_unpreferred_response_feedback(prompt_node)

                        next_level.extend(ptn.children)
                        n_prompt_refinement = max(self.n_prompt_refinemnet_children - len(ptn.children), 0)
                        next_prompts = self.refine_prompt(prompt_node, n_prompt_refinement)
                        for next_prompt_node in next_prompts:
                            child = TreeNode(next_prompt_node, parent=ptn)
                            ptn.children.append(child)
                            next_level.append(child)
                self.save_result(self.root, depth)
                depth += 1
                current_level = next_level
        return self.root

    def generate_queries(self, prompt_node: PromptNode) -> PromptNode:
        n_to_be_generated_queries = max(self.n_queries - len(prompt_node.queries), 0)
        queries = generate_queries(prompt_node, self.generate_model, self.budget, self.products, n_to_be_generated_queries)
        for query, product, attribute in queries:
            prompt_node.queries.append(QueryNode(query, product, attribute))
        return prompt_node

    def get_query_filter_feedback(self, prompt_node: PromptNode) -> PromptNode:
        res = query_filter_feedback(prompt_node, self.refinement_model, self.budget, self.filtering_policy)
        reason, improvement_suggestions = res

        prompt_node.query_filter_reason = reason
        prompt_node.query_filter_improvement_suggestions = improvement_suggestions

        return prompt_node

    def get_unpreferred_response_feedback(self, prompt_node: PromptNode) -> PromptNode:
        res = unpreferred_response_feedback(prompt_node, self.refinement_model, self.budget, self.unpreferred_definition)
        reason, improvement_suggestions = res

        prompt_node.unpreferred_response_reason = reason
        prompt_node.unpreferred_response_improvement_suggestions = improvement_suggestions

        return prompt_node

    def refine_prompt(self, prompt_node: PromptNode, n_prompt_refinement: int) -> List[PromptNode]:
        new_promtps = refine_prompt(
            prompt_node,
            self.refinement_model,
            self.budget,
            self.filtering_policy,
            self.preferred_definition,
            self.unpreferred_definition,
            n_prompt_refinement
        )
        
        next_prompt_nodes = []
        for prompt, improvement_reason in new_promtps:
            new_prompt_node = PromptNode(prompt, improvement_reason)
            next_prompt_nodes.append(new_prompt_node)

        return next_prompt_nodes
    
    def prune_prompt_nodes(self, prompt_tree_nodes: List[TreeNode]) -> List[TreeNode]:
        selected_tree_nodes = [ptn for ptn in prompt_tree_nodes if ptn.value.is_selected]
        not_selected_tree_nodes = [ptn for ptn in prompt_tree_nodes if not ptn.value.is_selected]

        beam = self.n_prompt_refinemnet_beam_search_size

        if len(selected_tree_nodes) >= beam:
            return selected_tree_nodes
        
        beam = beam - len(selected_tree_nodes)

        if len(sorted_nodes) <= beam:
            return selected_tree_nodes + not_selected_tree_nodes
        
        sorted_nodes = sorted(not_selected_tree_nodes, key=lambda n: n.value.score)
        
        cutoff_score = sorted_nodes[beam - 1].value.score

        strictly_better = [n for n in sorted_nodes if n.value.score < cutoff_score]
        tied_group = [n for n in sorted_nodes if n.value.score == cutoff_score]

        need = beam - len(strictly_better)

        if need >= len(tied_group):
            return selected_tree_nodes + (strictly_better + tied_group)[:beam]
        chosen_from_ties = random.sample(tied_group, need)

        return selected_tree_nodes + strictly_better + chosen_from_ties


    def result_to_json(self, result: TreeNode, max_depth = 10**9+7):
        root = result
        if not root: return []
        res = []
        node_id = 0
        queue = deque([(root, None, 0)])
        while queue:
            cur_prompt_tree_node, parent_id, depth = queue.popleft()
            if depth > max_depth: continue
            prompt_node: PromptNode = cur_prompt_tree_node.value
            prompt_row = {**{"id": node_id, "parent_id": parent_id}, **prompt_node.to_dict()}
            res.append(prompt_row)

            for next_prompt_tree_node in cur_prompt_tree_node.children:
                queue.append((next_prompt_tree_node, node_id, depth + 1))

            node_id += 1
        
        return res
    
    def save_result(self, result: TreeNode, depth):
        result_json = self.result_to_json(result, depth)
        save_depth_path = self.base_depth_output_path.with_stem(self.base_depth_output_path.stem + f'_depth{depth}')
        os.makedirs(save_depth_path.parent, exist_ok=True)
        with open(save_depth_path, 'w') as f:
            json.dump(result_json, f, indent=4)
        print(f"Save prompt search result: depth {depth}, to {save_depth_path}")

    def load_prev_result(self, result: list) -> "PromptSearch":
        product_map = {product["product_name"]: product for product in self.products}
        
        def laod_prev_queries(query_result: list) -> TreeNode:
            query_graph = defaultdict(list)
            for query in query_result:
                query_graph[query["parent_id"]].append(query)
            init_query = [query for query in query_result if query["parent_id"] == None][0]
            query_root = TreeNode(QueryNode("", {}, []))
            stack = [(query_root, init_query)]
            while stack:
                query_tree_node, query_json = stack.pop()
                query_node: QueryNode = query_tree_node.value
                query_node.query_refinement_strategy = query_json["query_refinement_strategy"]
                query_node.product = product_map[query_json["product"]]
                query_node.attributes = query_json["attributes"]
                query_node.query = query_json["query"]
                query_node.is_filtered_out = query_json["is_filtered_out"]
                query_node.filtering_reason = query_json["filtering_reason"]
                query_node.response = query_json["response"]
                query_node.individual_evaluations = query_json["individual_evaluations"]
                query_node.evaluation = query_json["evaluation"]
                query_node.evaluation_reason = query_json["evaluation_reason"]
                query_node.score = query_json["score"]
                query_node.set_is_unpreferred(query_json["evaluation"], self.unpreferred_definition)

                for next_query in query_graph.get(query_json["id"], []):
                    new_query_tree_node = TreeNode(QueryNode("", {}, []))
                    query_tree_node.children.append(new_query_tree_node)
                    stack.append((new_query_tree_node, next_query))
            return query_root


        prompt_graph = defaultdict(list)
        for prompt in result:
            prompt_graph[prompt["parent_id"]].append(prompt)
        init_prompt = [prompt for prompt in result if prompt["parent_id"] == None][0]
        self.root = TreeNode(PromptNode(""))
        stack = [(self.root, init_prompt)]
        while stack:
            prompt_tree_node, prompt_json = stack.pop()
            prompt_node: PromptNode = prompt_tree_node.value
            prompt_node.prompt = prompt_json["prompt"]
            prompt_node.improvement_reason = prompt_json["improvement_reason"]
            prompt_node.query_filter_reason = prompt_json["query_filter_reason"]
            prompt_node.query_filter_improvement_suggestions = prompt_json["query_filter_improvement_suggestions"]
            prompt_node.unpreferred_response_reason = prompt_json["unpreferred_response_reason"]
            prompt_node.unpreferred_response_improvement_suggestions = prompt_json["unpreferred_response_improvement_suggestions"]
            prompt_node.score = prompt_json["score"]
            for query_res in prompt_json["queries"]:
                query_tree_node = laod_prev_queries(query_res)
                prompt_node.queries.append(query_tree_node.value)
                prompt_node.query_refinement_map[query_tree_node.value] = query_tree_node

            for next_prompt in prompt_graph.get(prompt_json["id"], []):
                new_prompt_tree_node = TreeNode(PromptNode(""))
                prompt_tree_node.children.append(new_prompt_tree_node)
                stack.append((new_prompt_tree_node, next_prompt))

        
    