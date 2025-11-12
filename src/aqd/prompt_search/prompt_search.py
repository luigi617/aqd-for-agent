import json
import os
from typing import List, Optional
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from aqd.schema import PromptNode, QueryNode, TreeNode
from aqd.prompt_search.generate_queries import generate_queries
from aqd.prompt_search.refine_prompt import refine_prompt
from aqd.prompt_search.query_filter_feedback import query_filter_feedback
from aqd.prompt_search.unpreferred_response_feedback import unpreferred_response_feedback
from aqd.query_refinement.query_refinement import QueryRefinement


class PromptSearch:
    def __init__(self, prompt,
                 products,
                 n_queries,
                 generate_model,
                 refinement_model,
                 target_model,
                 filtering_model,
                 evaluation_model,
                 preferred_definition,
                 unpreferred_definition,
                 preference_score_map,
                 filtering_policy,

                 n_prompt_refinemnet_children,
                 n_prompt_refinemnet_beam_search_size,
                 n_query_refinemnet_iterations,
                 n_query_refiment_beam_search_size,
                 n_thread_queries,
                 n_thread_query_refiment,
                 base_depth_output_path):
        self.root: PromptNode = PromptNode(prompt)
        self.products: List[dict] = products
        self.n_queries: int = n_queries

        self.generate_model = generate_model
        self.refinement_model = refinement_model
        self.target_model = target_model
        self.filtering_model = filtering_model
        self.evaluation_model = evaluation_model

        self.preferred_definition = preferred_definition
        self.unpreferred_definition = unpreferred_definition
        self.preference_score_map = preference_score_map
        self.filtering_policy = filtering_policy

        self.n_prompt_refinemnet_children = n_prompt_refinemnet_children
        self.n_prompt_refinemnet_beam_search_size = n_prompt_refinemnet_beam_search_size
        self.n_query_refinemnet_iterations = n_query_refinemnet_iterations
        self.n_query_refiment_beam_search_size = n_query_refiment_beam_search_size
        self.n_thread_queries = n_thread_queries
        self.n_thread_query_refiment = n_thread_query_refiment

        self.base_depth_output_path = base_depth_output_path


    def _run_single_query_refinement(self, query_node):
        query_refinement = QueryRefinement(
            query_node=query_node,
            target_model=self.target_model,
            filtering_model=self.filtering_model,
            refinement_model=self.refinement_model,
            evaluation_model=self.evaluation_model,
            preferred_definition=self.preferred_definition,
            unpreferred_definition=self.unpreferred_definition,
            preference_score_map=self.preference_score_map,
            filtering_policy=self.filtering_policy,
            n_query_refiment_beam_search_size=self.n_query_refiment_beam_search_size,
        )
        return query_node, query_refinement.run(max_depth=self.n_query_refinemnet_iterations,
                                                n_workers=self.n_thread_query_refiment)

    def _process_node(self, tree_node, allow_refine: bool):
        prompt_node: PromptNode = tree_node.value
        with tqdm(total=5, desc="Processing prompt refinement node", unit="step") as pbar:
            prompt_node = self.generate_queries(prompt_node)
            pbar.update(1)
            max_workers = min(self.n_thread_queries, len(prompt_node.queries))
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as inner_pool:
                futures = [inner_pool.submit(self._run_single_query_refinement, q) for q in prompt_node.queries]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Refining query"):
                    query_node, query_tree_node = fut.result()
                    results.append((query_node, query_tree_node))
            pbar.update(1)
            for query_node, query_tree_node in results:
                prompt_node.query_refinement_map[query_node] = query_tree_node

            pbar.update(1)
            if not allow_refine:
                tree_node.value = prompt_node
                return tree_node, []
            
            prompt_node = self.get_query_filter_feedback(prompt_node)
            pbar.update(1)
            prompt_node = self.get_unpreferred_response_feedback(prompt_node)
            pbar.update(1)
            
            tree_node.value = prompt_node
            
            next_prompts = self.refine_prompt(prompt_node)
            pbar.update(1)
        
        
        return tree_node, next_prompts
    
    def run(self, max_depth = 3, budget = None, n_workers = 5):
        root_tree_node = TreeNode(self.root)

        current_level = [root_tree_node]
        depth = 0

        with ThreadPoolExecutor(max_workers = n_workers) as pool:
            while current_level:
                allow_refine = depth < max_depth
                futures = [
                    pool.submit(self._process_node, node, allow_refine)
                    for node in current_level
                ]

                next_level = []
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Prompt search: depth {depth}"):
                    tree_node, next_prompt_nodes = fut.result()

                    if allow_refine and next_prompt_nodes:
                        for prompt_node in next_prompt_nodes:
                            child = TreeNode(prompt_node, parent=tree_node)
                            tree_node.children.append(child)
                            next_level.append(child)
                self.save_result(root_tree_node, depth)
                depth += 1
                current_level = self.prune_prompt_nodes(next_level)
        return root_tree_node

    def generate_queries(self, prompt_node: PromptNode) -> PromptNode:
        queries = generate_queries(prompt_node, self.generate_model, self.products, self.n_queries)
        for query, product, attribute in queries:
            prompt_node.queries.append(QueryNode(query, product, attribute))
        return prompt_node

    def get_query_filter_feedback(self, prompt_node: PromptNode) -> PromptNode:
        res = query_filter_feedback(prompt_node, self.refinement_model, self.filtering_policy)
        reason, improvement_suggestions = res

        prompt_node.query_filter_reason = reason
        prompt_node.query_filter_improvement_suggestions = improvement_suggestions

        return prompt_node

    def get_unpreferred_response_feedback(self, prompt_node: PromptNode) -> PromptNode:
        res = unpreferred_response_feedback(prompt_node, self.refinement_model, self.unpreferred_definition)
        reason, improvement_suggestions = res

        prompt_node.unpreferred_response_reason = reason
        prompt_node.unpreferred_response_improvement_suggestions = improvement_suggestions

        return prompt_node

    def refine_prompt(self, prompt_node: PromptNode) -> List[PromptNode]:
        new_promtps = refine_prompt(
            prompt_node,
            self.refinement_model,
            self.filtering_policy,
            self.preferred_definition,
            self.unpreferred_definition,
            self.n_prompt_refinemnet_children
        )
        
        next_prompt_nodes = []
        for prompt, improvement_reason in new_promtps:
            new_prompt_node = PromptNode(prompt, improvement_reason)
            next_prompt_nodes.append(new_prompt_node)

        return next_prompt_nodes
    
    def prune_prompt_nodes(self, prompt_nodes: List[TreeNode]) -> List[TreeNode]:
        remaining_prompt_nodes = []
        # {prevn node : [cur nodes]}
        prompt_nodes_relation_map = defaultdict(list)
        for prompt_node in prompt_nodes:
            parent = prompt_node.parent
            prompt_nodes_relation_map[parent].append(prompt_node)
        
        parent_nodes = list(prompt_nodes_relation_map.keys())
        for parent in parent_nodes:
            parent.value.get_score(self.preference_score_map)
        remaining_parent_nodes = sorted(parent_nodes, key=lambda node: node.value)[:self.n_prompt_refinemnet_beam_search_size]
        for parent in remaining_parent_nodes:
            remaining_prompt_nodes.extend(prompt_nodes_relation_map[parent])

        return remaining_prompt_nodes


    def result_to_json(self, result: TreeNode):
        root = result
        if not root: return []
        res = []
        node_id = 0
        queue = deque([(root, None)])
        while queue:
            cur_prompt_tree_node, parent_id = queue.popleft()
            prompt_node: PromptNode = cur_prompt_tree_node.value
            prompt_row = {**{"id": node_id, "parent_id": parent_id}, **prompt_node.to_dict(self.preference_score_map)}
            res.append(prompt_row)

            for next_prompt_tree_node in cur_prompt_tree_node.children:
                queue.append((next_prompt_tree_node, node_id))

            node_id += 1
        return res
    
    def save_result(self, result: TreeNode, depth):
        result_json = self.result_to_json(result)
        save_depth_path = self.base_depth_output_path.with_stem(self.base_depth_output_path.stem + f'_depth{depth}')
        os.makedirs(save_depth_path.parent, exist_ok=True)
        with open(save_depth_path, 'w') as f:
            json.dump(result_json, f, indent=4)
        print(f"Save prompt search result: depth {depth}, to {save_depth_path}")

    
    