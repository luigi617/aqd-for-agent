
import random
from collections import defaultdict
import time
from tqdm import tqdm
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.aqd.ucb import UCB
from src.config import Config
from src.aqd.query_refinement.filter_queries import filter_queries
from src.aqd.query_refinement.generate_response import generate_response
from src.aqd.query_refinement.evaluate_responses import evaluate_responses
from src.aqd.schema import QueryNode, TreeNode
from src.aqd.query_refinement.refine_query import refine_query




class QueryRefinement:
    
    def __init__(self, query_node, query_tree_node, config: Config):
        self.config: Config = config

        if isinstance(query_tree_node, TreeNode):
            self.root: TreeNode = query_tree_node
        else:
            self.root: TreeNode = TreeNode(query_node)
        
            
        if self.root.value.strategy_ucb is None:
            self.root.value.strategy_ucb = UCB(
                arms=list(self.config.query_refinement_strategies.keys()),
                variant="ucb1",
                epsilon=0.10,
            )
        

    def _process_node(self, tree_node: TreeNode):
        query_node: QueryNode = tree_node.value
        start_time = time.perf_counter()

        with tqdm(total=4, desc="Processing query refinement node", unit="step") as pbar:
            strategy = query_node.query_refinement_strategy
            query_node = self.filter_queries(query_node)
            pbar.update(1)
            if query_node.is_filtered_out:
                end_time = time.perf_counter()
                query_node.set_running_time(end_time - start_time)
                return tree_node, []

            query_node = self.generate_response(query_node)
            pbar.update(1)
            query_node = self.evaluate_responses(query_node)
            pbar.update(1)
            query_node = self.modify_responses(query_node)
            pbar.update(1)
            query_node_score = query_node.get_score(self.config.preference_score_map)
            
            end_time = time.perf_counter()
            query_node.set_running_time(end_time - start_time)

            if strategy is not None and tree_node.parent:
                parent_query_node_score = tree_node.parent.value.get_score(self.config.preference_score_map)
                if isinstance(parent_query_node_score, float) and isinstance(query_node_score, float):
                    reward = max(parent_query_node_score - query_node_score, 0)
                    self.root.value.strategy_ucb.update(strategy, reward)

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
                    try:
                        fut.result()
                    except Exception as e:
                        pool.shutdown(wait=False, cancel_futures=True)
                        raise
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
        satisfy_filter, reasoning, filtering_confidence_score = filter_queries(query_node,
                                                   self.config.filtering_model,
                                                   self.config.budget,
                                                   self.config.filtering_policy)
        query_node.is_filtered_out = not satisfy_filter
        query_node.filtering_reason = reasoning
        query_node.filtering_confidence_score = filtering_confidence_score
        return query_node
    
    def generate_response(self, query_node: QueryNode) -> QueryNode:
        response = generate_response(query_node, self.config.target_model, self.config.budget)
        query_node.response = response
        return query_node
    
    def evaluate_responses(self, query_node: QueryNode) -> QueryNode:
        criteria_chosen, reason, individual_evals = evaluate_responses(query_node,
                                                     self.config.evaluation_models,
                                                     self.config.refinement_model,
                                                     self.config.budget,
                                                     self.config.preferred_definition,
                                                     self.config.unpreferred_definition)
        
        query_node.set_is_unpreferred(criteria_chosen, self.config.unpreferred_definition)
        query_node.evaluation = criteria_chosen
        query_node.evaluation_reason = reason
        query_node.individual_evaluations = individual_evals
        return query_node
    
    def modify_responses(self, query_node: QueryNode) -> QueryNode:
        # TODO: implement me
        return query_node
    
    def refine_query(self, tree_node: TreeNode) -> List[QueryNode]:
        strategy_used = [child_tree_node.value.query_refinement_strategy for child_tree_node in tree_node.children]
        
        current_strategy_apply_count = defaultdict(int)
        strategy_mutex_group = set()
        cur_qtn = tree_node
        while cur_qtn:
            strategy = cur_qtn.value.query_refinement_strategy
            if strategy is None: break
            mutex_group = self.config.query_refinement_strategies[strategy]["mutex_group"]
            if mutex_group is not None:
                strategy_mutex_group.add(mutex_group)
            current_strategy_apply_count[strategy] += 1
            cur_qtn = cur_qtn.parent

        strategy_not_allowed = []
        for strategy, info in self.config.query_refinement_strategies.items():
            if current_strategy_apply_count[strategy] >= info["max_apply_num"]:
                strategy_not_allowed.append(strategy)
            if info["mutex_group"] in strategy_mutex_group:
                strategy_not_allowed.append(strategy)

        strategies_ignored = list(set(strategy_used + strategy_not_allowed))

        strategy_size = 3
        n_strategy_to_select = min(strategy_size, len(self.config.query_refinement_strategies) - len(strategies_ignored))
        strategy_to_be_used = self.root.value.strategy_ucb.select(k=n_strategy_to_select,
                                                                  strategies_ignored=strategies_ignored,
                                                                  try_all=tree_node.parent is None)
        
        query_node: QueryNode = tree_node.value
        proposed_queries: List[Tuple[str, str]] = refine_query(
            tree_node,
            self.config.refinement_model,
            self.config.budget,
            strategy_to_be_used,
            self.config.query_refinement_strategies,
        )
        next_query_nodes = []
        for query, strategy in proposed_queries:
            new_query_node = QueryNode(query, query_generation_extra_info=query_node.query_generation_extra_info, query_refinement_strategy=strategy)
            next_query_nodes.append(new_query_node)
        return next_query_nodes
    
    def _prune_query_nodes(self, query_tree_nodes: List[TreeNode]) -> List[TreeNode]:
        valid_nodes = [n for n in query_tree_nodes if n.value.score is not None]
        selected_tree_nodes = [n for n in valid_nodes if n.children]
        not_selected_tree_nodes = [n for n in valid_nodes if not n.children]

        beam = self.config.n_query_refiment_beam_search_size

        if len(selected_tree_nodes) >= beam:
            return selected_tree_nodes
        
        beam = beam - len(selected_tree_nodes)

        if len(not_selected_tree_nodes) <= beam:
            return selected_tree_nodes + not_selected_tree_nodes

        sorted_nodes = sorted(not_selected_tree_nodes, key=lambda node: node.value.score)

        cutoff_score = sorted_nodes[beam - 1].value.score

        strictly_better = [n for n in sorted_nodes if n.value.score < cutoff_score]
        tied_group = [n for n in sorted_nodes if n.value.score == cutoff_score]

        need = beam - len(strictly_better)

        if need >= len(tied_group):
            # Take all ties, truncate if slightly over
            return selected_tree_nodes + (strictly_better + tied_group)[:beam]
    
        chosen_from_ties = random.sample(tied_group, need)

        return selected_tree_nodes + strictly_better + chosen_from_ties
    

    def prune_query_nodes(self, query_tree_nodes: List[TreeNode]) -> List[TreeNode]:
        remaining_query_tree_nodes = self._prune_query_nodes(query_tree_nodes)
        min_score = min(self.config.preference_score_map.values())
        return [qtn for qtn in remaining_query_tree_nodes if qtn.value.get_score(self.config.preference_score_map) > min_score]