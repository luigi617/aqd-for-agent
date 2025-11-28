import contextvars
import random
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
                 refinement_model, evaluation_models,
                 preferred_definition,
                 unpreferred_definition,
                 preference_score_map,
                 filtering_policy,
                 query_refinement_strategies,
                 n_query_refiment_beam_search_size,
                 budget,):
        
        self.root: TreeNode = TreeNode(query_node)

        self.target_model = target_model
        self.filtering_model = filtering_model
        self.refinement_model = refinement_model
        self.evaluation_models = evaluation_models

        self.preferred_definition = preferred_definition
        self.unpreferred_definition = unpreferred_definition
        self.preference_score_map = preference_score_map
        self.filtering_policy = filtering_policy
        self.query_refinement_strategies = query_refinement_strategies
        
        self.n_query_refiment_beam_search_size = n_query_refiment_beam_search_size
        self.budget = budget

    def _process_node(self, tree_node):
        query_node = tree_node.value
        with tqdm(total=4, desc="Processing query refinement node", unit="step") as pbar:

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
            query_node.get_score(self.preference_score_map)

            tree_node.value = query_node

        return tree_node

    def run(self, max_depth = 3, n_workers = 5):
        """
            TAP algorithm
            https://arxiv.org/pdf/2312.02119
        """

        current_level = [self.root]
        depth = 0

        with ThreadPoolExecutor(max_workers = n_workers) as pool:
            while current_level:
                allow_refine = depth < max_depth
                futures = [
                    pool.submit(self._process_node, node)
                    for node in current_level
                ]

                next_level = []

                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Query refinement: depth {depth}"):
                    fut.result()
                if allow_refine:
                    query_tree_node_to_be_rifined = self.prune_query_nodes(current_level)
                    for qtn in query_tree_node_to_be_rifined:
                        next_queries = self.refine_query(qtn)
                        for q in next_queries:
                            child = TreeNode(q, parent=qtn)
                            qtn.children.append(child)
                            next_level.append(child)

                depth += 1
                current_level = next_level
        
        return self.root
            
        
    
    def filter_queries(self, query_node: QueryNode) -> QueryNode:
        satisfy_filter, reasoning = filter_queries(query_node,
                                                   self.filtering_model,
                                                   self.budget,
                                                   self.filtering_policy)
        query_node.is_filtered_out = not satisfy_filter
        query_node.filtering_reason = reasoning
        return query_node
    
    def generate_response(self, query_node: QueryNode) -> QueryNode:
        response = generate_response(query_node, self.target_model, self.budget,)
        query_node.response = response
        return query_node
    
    def evaluate_responses(self, query_node: QueryNode) -> QueryNode:
        criteria_chosen, reason, individual_evals = evaluate_responses(query_node,
                                                     self.evaluation_models,
                                                     self.refinement_model,
                                                     self.budget,
                                                     self.preferred_definition,
                                                     self.unpreferred_definition)
        
        query_node.set_is_unpreferred(criteria_chosen, self.unpreferred_definition)
        query_node.evaluation = criteria_chosen
        query_node.evaluation_reason = reason
        query_node.individual_evaluations = individual_evals
        return query_node
    
    def modify_responses(self, query_node: QueryNode) -> QueryNode:
        # TODO: implement me
        return query_node
    
    def refine_query(self, tree_node: TreeNode) -> List[QueryNode]:
        query_node = tree_node.value
        proposed_queries: List[Tuple[str, str]] = refine_query(
            tree_node,
            self.refinement_model,
            self.budget,
            self.query_refinement_strategies
        )
        next_query_nodes = []
        for query, strategy in proposed_queries:
            new_query_node = QueryNode(query, product=query_node.product, attributes=query_node.attributes, query_refinement_strategy=strategy)
            next_query_nodes.append(new_query_node)
        return next_query_nodes
    
    def prune_query_nodes(self, query_tree_nodes: List[TreeNode]) -> List[TreeNode]:
        valid_nodes = [n for n in query_tree_nodes if n.value.score is not None]

        beam = self.n_query_refiment_beam_search_size
        if len(valid_nodes) <= beam:
            return valid_nodes

        sorted_nodes = sorted(valid_nodes, key=lambda node: node.value.score)

        cutoff_score = sorted_nodes[beam - 1].value.score

        strictly_better = [n for n in sorted_nodes if n.value.score < cutoff_score]
        tied_group = [n for n in sorted_nodes if n.value.score == cutoff_score]

        need = beam - len(strictly_better)

        # Safety guards (cheap + avoids the ValueError)
        if need <= 0:
            return strictly_better[:beam]

        if need >= len(tied_group):
            # Take all ties, truncate if slightly over
            return (strictly_better + tied_group)[:beam]
    
        chosen_from_ties = random.sample(tied_group, need)

        return strictly_better + chosen_from_ties
    

