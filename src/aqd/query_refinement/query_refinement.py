from collections import defaultdict, deque
from tqdm import tqdm
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from aqd.query_refinement.filter_queries import filter_queries
from aqd.query_refinement.generate_response import generate_response
from aqd.query_refinement.evaluate_responses import evaluate_responses
from aqd.schema import QueryNode, TreeNode
from aqd.query_refinement.refine_query import refine_query




class QueryRefinement:
    
    def __init__(self, query_node, target_model,
                 filtering_model,
                 refinement_model, evaluation_model,
                 preferred_definition,
                 unpreferred_definition,
                 preference_score_map,
                 filtering_policy,
                 n_query_refiment_beam_search_size):
        
        self.root: QueryNode = query_node

        self.target_model = target_model
        self.filtering_model = filtering_model
        self.refinement_model = refinement_model
        self.evaluation_model = evaluation_model

        self.preferred_definition = preferred_definition
        self.unpreferred_definition = unpreferred_definition
        self.preference_score_map = preference_score_map
        self.filtering_policy = filtering_policy
        
        self.n_query_refiment_beam_search_size = n_query_refiment_beam_search_size

    def _process_node(self, tree_node, allow_refine: bool):
        query_node = tree_node.value
        with tqdm(total=5, desc="Processing query refinement node", unit="step") as pbar:

            query_node = self.filter_queries(query_node)
            pbar.update(1)
            if query_node.is_filtered_out:
                tree_node.value = query_node  # keep the filtered state
                return tree_node, []

            query_node = self.generate_response(query_node)
            pbar.update(1)
            query_node = self.evaluate_responses(query_node)
            pbar.update(1)
            query_node = self.modify_responses(query_node)
            pbar.update(1)

            tree_node.value = query_node

            if not allow_refine:
                return tree_node, []

            next_queries = self.refine_query(tree_node)
            pbar.update(1)
        return tree_node, next_queries

    def run(self, max_depth = 3, n_workers = 5):
        """
            TAP algorithm
            https://arxiv.org/pdf/2312.02119
        """
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
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Query refinement: depth {depth}"):
                    tree_node, next_queries = fut.result()

                    if allow_refine and next_queries:
                        for q in next_queries:
                            child = TreeNode(q, parent=tree_node)
                            tree_node.children.append(child)
                            next_level.append(child)

                depth += 1
                current_level = self.prune_query_nodes(next_level)
        
        return root
            
        
    
    def filter_queries(self, query_node: QueryNode) -> QueryNode:
        satisfy_filter, reasoning = filter_queries(query_node,
                                                   self.filtering_model,
                                                   self.filtering_policy)
        query_node.is_filtered_out = not satisfy_filter
        query_node.filtering_reason = reasoning
        return query_node
    
    def generate_response(self, query_node: QueryNode) -> QueryNode:
        response = generate_response(query_node, self.target_model)
        query_node.response = response
        return query_node
    
    def evaluate_responses(self, query_node: QueryNode) -> QueryNode:
        criteria_chosen, reason = evaluate_responses(query_node,
                                                     self.target_model,
                                                     self.preferred_definition,
                                                     self.unpreferred_definition)
        query_node.is_unpreferred = False
        for criterion, _ in criteria_chosen:
            if criterion in self.unpreferred_definition:
                query_node.is_unpreferred = True
                break
        query_node.evaluation = criteria_chosen
        query_node.evaluation_reason = reason
        return query_node
    
    def modify_responses(self, query_node: QueryNode) -> QueryNode:
        # TODO: implement me
        return query_node
    
    def refine_query(self, tree_node: TreeNode) -> List[QueryNode]:
        query_node = tree_node.value
        proposed_queries: List[Tuple[str, str]] = refine_query(tree_node, self.refinement_model)
        next_query_nodes = []
        for query, strategy in proposed_queries:
            new_query_node = QueryNode(query, product=query_node.product, attributes=query_node.attributes, query_refinement_strategy=strategy)
            next_query_nodes.append(new_query_node)
        return next_query_nodes
    
    def prune_query_nodes(self, query_nodes: List[TreeNode]) -> List[TreeNode]:
        remaining_query_nodes = []
        # {prev node : [cur nodes]}
        query_nodes_relation_map = defaultdict(list)
        for query_node in query_nodes:
            parent = query_node.parent
            query_nodes_relation_map[parent].append(query_node)
        
        parent_nodes = list(query_nodes_relation_map.keys())
        for parent in parent_nodes:
            parent.value.get_score(self.preference_score_map)
        remaining_parent_nodes = sorted(parent_nodes, key=lambda node: node.value)[:self.n_query_refiment_beam_search_size]
        for parent in remaining_parent_nodes:
            remaining_query_nodes.extend(query_nodes_relation_map[parent])

        return remaining_query_nodes
    

