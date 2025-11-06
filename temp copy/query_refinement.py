from collections import deque
from typing import List, Tuple

class QueryNode:
    def __init__(self, query: str):
        self.query: str = query
        
        self.is_filtered_out: bool = False
        self.filtering_reason: str = None
        
        self.response: str = None

        self.evaluation: List(Tuple(str, str)) = []
        pass

class QueryRefinement:
    
    def __init__(self, query):
        self.root = QueryNode(query)
        pass

    def run(self):
        """
            TAP algorithm
            https://arxiv.org/pdf/2312.02119
        """
        queue = deque([self.root])

        while queue:
            query_node = queue.popleft()
            query_node = self.filter_queries(query_node)
            if query_node.is_filtered_out: # query does not satisfy filterint policy
                continue
            query_node = self.generate_response(query_node)
            query_node = self.evaluate_responses(query_node)
            query_node = self.modify_responses(query_node)
            next_queries = self.refine_query(query_node)
            for next_query in next_queries:
                queue.append(next_query)
        
    
    def filter_queries(self, query_node: QueryNode) -> QueryNode:
        # TODO: implement me
        return query_node
    
    def generate_response(self, query_node: QueryNode) -> QueryNode:
        # TODO: implement me
        return query_node
    
    def evaluate_responses(self, query_node: QueryNode) -> QueryNode:
        # TODO: implement me
        return query_node
    
    def modify_responses(self, query_node: QueryNode) -> QueryNode:
        # TODO: implement me
        return query_node
    
    def refine_query(self, query_node: QueryNode) -> List(QueryNode):
        # TODO: implement me
        return []
    

