from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from aqd.schema import PromptNode, QueryNode, TreeNode
from aqd.prompt_search.generate_queries import generate_queries
from src.aqd.prompt_search.refine_prompt import refine_prompt
from src.aqd.prompt_search.query_filter_feedback import query_filter_feedback
from src.aqd.prompt_search.unpreferred_response_feedback import unpreferred_response_feedback
from src.aqd.query_refinement.query_refinement import QueryRefinement


class PromptSearch:
    def __init__(self, prompt, generate_model, refinement_model,
                 target_model,
                 filtering_model,
                 evaluation_model,
                 preferred_definition,
                 unpreferred_definition,
                 filtering_policy):
        self.root: PromptNode = PromptNode(prompt)

        self.generate_model = generate_model
        self.refinement_model = refinement_model
        self.target_model = target_model
        self.filtering_model = filtering_model
        self.evaluation_model = evaluation_model

        self.preferred_definition = preferred_definition
        self.unpreferred_definition = unpreferred_definition
        self.filtering_policy = filtering_policy

    def _process_node(self, tree_node, allow_refine: bool):
        prompt_node: PromptNode = tree_node.value

        prompt_node = self.generate_queries(prompt_node)

        for query_node in prompt_node.queries:
            query_refinement = QueryRefinement(
                query_node,
                self.target_model,
                self.filtering_model,
                self.refinement_model,
                self.evaluation_model,
                self.preferred_definition,
                self.unpreferred_definition,
                self.filtering_policy
            )
            query_tree_node = query_refinement.run()
            prompt_node.query_refinement_map[query_node] = query_tree_node

        if not allow_refine:
            tree_node.value = prompt_node
            return tree_node, []
        
        prompt_node = self.get_query_filter_feedback(prompt_node)
        prompt_node = self.get_unpreferred_response_feedback(prompt_node)
        tree_node.value = prompt_node

        next_prompts = self.refine_prompt(prompt_node)
        return tree_node, next_prompts
    
    def run(self, max_depth = 3, budget = None, n_workers = 5):
        root = TreeNode(self.root)

        current_level = [root]
        depth = 0

        with ThreadPoolExecutor(max_workers = n_workers) as pool:
            while current_level:
                allow_refine = depth < max_depth

                futures = [
                    pool.submit(self._process_node, node, allow_refine)
                    for node in current_level
                ]

                next_level = []
                for fut in as_completed(futures):
                    tree_node, next_prompts = fut.result()

                    if allow_refine and next_prompts:
                        for p in next_prompts:
                            child = TreeNode(p, parent=tree_node)
                            tree_node.children.append(child)
                            next_level.append(child)

                depth += 1
                current_level = next_level
        return root

    def generate_queries(self, prompt_node: PromptNode) -> PromptNode:
        queries = generate_queries(prompt_node, self.generate_model)
        for query, product, attribute in queries:
            prompt_node.queries.append(QueryNode(query, product, attribute))
        return prompt_node

    def get_query_filter_feedback(self, prompt_node: PromptNode) -> PromptNode:
        res = query_filter_feedback(prompt_node, self.refinement_model, self.filtering_policy)
        reason, contributing_prompt_patterns, improvement_suggestions = res

        prompt_node.query_filter_reason = reason
        prompt_node.query_filter_contributing_prompt_patterns = contributing_prompt_patterns
        prompt_node.query_filter_improvement_suggestions = improvement_suggestions

        return prompt_node

    def get_unpreferred_response_feedback(self, prompt_node: PromptNode) -> PromptNode:
        res = unpreferred_response_feedback(prompt_node, self.refinement_model, self.filtering_policy)
        reason, contributing_prompt_patterns, improvement_suggestions = res

        prompt_node.unpreferred_response_reason = reason
        prompt_node.unpreferred_response_contributing_prompt_patterns = contributing_prompt_patterns
        prompt_node.unpreferred_response_improvement_suggestions = improvement_suggestions

        return prompt_node

    def refine_prompt(self, prompt_node: PromptNode) -> List[PromptNode]:
        new_promtps = refine_prompt(prompt_node, self.refinement_model, 3)
        
        next_prompt_nodes = []
        for prompt, reason in new_promtps:
            new_prompt_node = PromptNode(prompt, reason)
            next_prompt_nodes.append(new_prompt_node)

        return next_prompt_nodes

