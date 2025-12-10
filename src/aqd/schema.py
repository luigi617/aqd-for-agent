from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from src.aqd.ucb import UCB

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

        self.score: Optional[float] = None


    def to_dict(self) -> dict:
        return {
            "improvement_reason": self.improvement_reason,
            "prompt": self.prompt,
            "queries": [self.query_to_json(self.query_refinement_map[query]) for query in self.queries],
            
            "query_filter_reason": self.query_filter_reason,
            "query_filter_improvement_suggestions": self.query_filter_improvement_suggestions,

            "unpreferred_response_reason": self.unpreferred_response_reason,
            "unpreferred_response_improvement_suggestions": self.unpreferred_response_improvement_suggestions,
            "score": self.score
        }
    
    @property
    def is_selected(self):
        return self.query_filter_reason != "" or \
            self.query_filter_reason != "" or \
                self.query_filter_reason != "" or \
                self.query_filter_reason != ""
    
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
    
    def get_score(self, preference_score_map: dict):
        if self.score is not None:
            return self.score
        count = 0
        score = 0
        for query_tree_node in self.query_refinement_map.values():
            stack = [query_tree_node]
            while stack:
                cur_query_tree_node = stack.pop()
                cur_query_node = cur_query_tree_node.value
                if not cur_query_node.is_filtered_out:
                    query_score = cur_query_node.get_score(preference_score_map)
                    if query_score is not None:
                        score += query_score
                        count += 1
                stack.extend(cur_query_tree_node.children)
        self.score = score / count if count > 0 else float("inf")
        return self.score
    
    
    def __gt__(self, other: "PromptNode"):
        return self.score > other.score
    
    def __ge__(self, other: "PromptNode"):
        return self.score >= other.score
    
    def __lt__(self, other: "PromptNode"):
        return self.score < other.score
    
    def __le__(self, other: "PromptNode"):
        return self.score <= other.score
    

class QueryNode:
    def __init__(self, query: str, query_generation_extra_info: dict, query_refinement_strategy: Optional[str] = None):
        self.query_refinement_strategy: Optional[str] = query_refinement_strategy
        
        self.query_generation_extra_info: dict = query_generation_extra_info
        
        self.query: str = query
        
        self.is_filtered_out: bool = False
        self.filtering_reason: str = ""
        
        self.response: str = ""

        self.individual_evaluations: List[Any] = []

        self.evaluation: List[Tuple[str, str]] = []
        self.evaluation_reason: str = ""

        self.is_unpreferred: Optional[bool] = None

        self.score: Optional[float] = None

        self.strategy_ucb: Optional[UCB] = None

    def set_is_unpreferred(self, criteria_chosen, unpreferred_definition) -> None:
        is_unpreferred = False
        for criterion, _ in criteria_chosen:
            if criterion in unpreferred_definition:
                is_unpreferred = True
        self.is_unpreferred = is_unpreferred

    def get_score(self, preference_score_map: dict):
        if self.is_filtered_out:
            return None
        if self.evaluation is None or len(self.evaluation) == 0:
            return None
        if self.score is not None:
            return self.score
        score = sum(preference_score_map[criterion] for criterion, _ in self.evaluation) / len(self.evaluation)
        self.score = score
        return score
    
    def __gt__(self, other: "QueryNode"):
        return self.score > other.score
    
    def __ge__(self, other: "QueryNode"):
        return self.score >= other.score
    
    def __lt__(self, other: "QueryNode"):
        return self.score < other.score
    
    def __le__(self, other: "QueryNode"):
        return self.score <= other.score


    def to_dict(self) -> dict:
        query_generation_extra_info = self.query_generation_extra_info.copy()
        if "product" in query_generation_extra_info:
            query_generation_extra_info["product"] = query_generation_extra_info["product"].get("product_name")
        return {
            "query": self.query,
            "query_generation_extra_info": query_generation_extra_info,
            "is_filtered_out": self.is_filtered_out,
            "filtering_reason": self.filtering_reason,
            "response": self.response,
            "individual_evaluations": self.individual_evaluations,
            "evaluation": [list(ev) for ev in self.evaluation],
            "evaluation_reason": self.evaluation_reason,
            "query_refinement_strategy": self.query_refinement_strategy,
            "score": self.score,
            "strategy_ucb": self.strategy_ucb.to_dict() if isinstance(self.strategy_ucb, UCB) else None,
        }


class TreeNode:
    def __init__(self,
                 value: Any,
                 parent: Optional["TreeNode"] = None,
                 children: Optional[List["TreeNode"]] = None):
        
        self.value = value
        self.parent = parent
        self.children = children if children is not None else []

    