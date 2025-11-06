from collections import deque
from typing import List, Optional, Tuple

from aqd.query_refinement.filter_queries import filter_queries
from aqd.query_refinement.generate_response import generate_response
from aqd.query_refinement.evaluate_responses import evaluate_responses
from aqd.schema import QueryNode

class TreeNode:
    def __init__(self, value: QueryNode, children: Optional[List["TreeNode"]] = None):
        self.value = value
        self.children = children if children is not None else []

class QueryRefinement:
    
    def __init__(self, query_node, target_model,
                 filtering_model,
                 refinement_model, evaluation_model,
                 preferred_definition,
                 unpreferred_definition,
                 filtering_policy):
        
        self.root: QueryNode = query_node

        self.target_model = target_model
        self.filtering_model = filtering_model
        self.refinement_model = refinement_model
        self.evaluation_model = evaluation_model

        self.preferred_definition = preferred_definition
        self.unpreferred_definition = unpreferred_definition
        self.filtering_policy = filtering_policy
        pass

    def run(self, max_depth = 3, budget = None):
        """
            TAP algorithm
            https://arxiv.org/pdf/2312.02119
        """
        root = TreeNode(self.root)

        queue = deque([(root, 0)])


        while queue:
            tree_node, depth = queue.popleft()
            query_node = tree_node.value

            query_node = self.filter_queries(query_node)
            if query_node.is_filtered_out: # query does not satisfy filterint policy
                continue
            query_node = self.generate_response(query_node)
            query_node = self.evaluate_responses(query_node)
            query_node = self.modify_responses(query_node)

            if depth >= max_depth:
                continue

            next_queries = self.refine_query(query_node)
            for next_query in next_queries:
                new_tree_node = TreeNode(next_query)
                tree_node.children.append(new_tree_node)
                queue.append((new_tree_node, depth + 1))
        
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
        query_node.evaluation = criteria_chosen
        query_node.evaluation_reason = reason
        return query_node
    
    def modify_responses(self, query_node: QueryNode) -> QueryNode:
        # TODO: implement me
        return query_node
    
    def refine_query(self, query_node: QueryNode) -> List[QueryNode]:
        # TODO: implement me
        return []
    

