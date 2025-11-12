from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from functools import cached_property

class PromptNode:
    def __init__(self, prompt: str, improvement_reason: Optional[str] = None):
        self.improvement_reason: Optional[str] = improvement_reason
        self.prompt: str = prompt
        self.queries: List[QueryNode] = []

        self.query_refinement_map: Dict[QueryNode, TreeNode] = {}

        self.query_filter_reason: str = ""
        self.query_filter_improvement_suggestions: str = ""

        self.unpreferred_response_reason: str = ""
        self.unpreferred_response_improvement_suggestions: str = ""

    def to_dict(self) -> dict:
        return {
            "improvement_reason": self.improvement_reason,
            "prompt": self.prompt,
            "queries": [self.query_to_json(self.query_refinement_map[query]) for query in self.queries],
            
            "query_filter_reason": self.query_filter_reason,
            "query_filter_improvement_suggestions": self.query_filter_improvement_suggestions,

            "unpreferred_response_reason": self.unpreferred_response_reason,
            "unpreferred_response_improvement_suggestions": self.unpreferred_response_improvement_suggestions,
            "score": self._compute_score
        }
    
    def query_to_json(self, query_tree_node: "TreeNode"):
        if not query_tree_node: return []
        res = []
        node_id = 0
        queue = deque([(query_tree_node, None)])
        while queue:
            cur_query_tree_node, parent_id = queue.popleft()
            query_node: QueryNode = cur_query_tree_node.value
            prompt_row = {**{"id": node_id, "parent_id": parent_id}, **query_node.to_dict()}
            res.append(prompt_row)

            for next_query_tree_node in cur_query_tree_node.children:
                queue.append((next_query_tree_node, node_id))

            node_id += 1
        return res
    
    @cached_property
    def _compute_score(self):
        score = 0
        for query_tree_node in self.query_refinement_map.values():
            stack = [query_tree_node]
            while stack:
                cur_query_tree_node = stack.pop()
                score += (cur_query_tree_node.value.is_unpreferred == True)
                stack.extend(cur_query_tree_node.children)
        return score
    
    def __gt__(self, other: "PromptNode"):
        return self._compute_score > other._compute_score
    
    def __ge__(self, other: "PromptNode"):
        return self._compute_score >= other._compute_score
    
    def __lt__(self, other: "PromptNode"):
        return self._compute_score < other._compute_score
    
    def __le__(self, other: "PromptNode"):
        return self._compute_score <= other._compute_score
    


class QueryNode:
    def __init__(self, query: str, product: dict, attributes: List, query_refinement_strategy: Optional[str] = None):
        self.query_refinement_strategy: Optional[str] = query_refinement_strategy
        
        self.product = product
        self.attributes = attributes
        
        self.query: str = query
        
        self.is_filtered_out: bool = False
        self.filtering_reason: str = ""
        
        self.response: str = ""

        self.evaluation: List[Tuple[str, str]] = []
        self.evaluation_reason: str = ""

        self.is_unpreferred: Optional[bool] = None

    @cached_property
    def _compute_score(self):
        # TODO: improvement me
        if self.is_filtered_out is None:
            return float("-inf")
        return int(self.is_unpreferred)
    
    def __gt__(self, other: "PromptNode"):
        return self._compute_score > other._compute_score
    
    def __ge__(self, other: "PromptNode"):
        return self._compute_score >= other._compute_score
    
    def __lt__(self, other: "PromptNode"):
        return self._compute_score < other._compute_score
    
    def __le__(self, other: "PromptNode"):
        return self._compute_score <= other._compute_score


    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "product": self.product.get("product_name"),
            "attributes": self.attributes,
            "is_filtered_out": self.is_filtered_out,
            "filtering_reason": self.filtering_reason,
            "response": self.response,
            "evaluation": [list(ev) for ev in self.evaluation],
            "evaluation_reason": self.evaluation_reason,
            "query_refinement_strategy": self.query_refinement_strategy,
            "score": self._compute_score,
        }


class TreeNode:
    def __init__(self,
                 value: Any,
                 parent: Optional["TreeNode"] = None,
                 children: Optional[List["TreeNode"]] = None):
        
        self.value = value
        self.parent = parent
        self.children = children if children is not None else []

    