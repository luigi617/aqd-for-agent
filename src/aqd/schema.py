from typing import Any, Dict, List, Optional, Tuple

class PromptNode:
    def __init__(self, prompt: str, reason: Optional[str] = None):
        self.reason: Optional[str] = reason
        self.prompt: str = prompt
        self.queries: List[QueryNode] = []

        self.query_refinement_map: Dict[QueryNode, TreeNode] = {}

        self.query_filter_reason: str = ""
        self.query_filter_contributing_prompt_patterns: str = ""
        self.query_filter_improvement_suggestions: str = ""

        self.unpreferred_response_reason: str = ""
        self.unpreferred_response_contributing_prompt_patterns: str = ""
        self.unpreferred_response_improvement_suggestions: str = ""

    def to_dict(self) -> dict:
        return {
            "prompt": self.query,
            "queries": [query.to_dict() for query in self.queries],
        }

class QueryNode:
    def __init__(self, query: str, product: dict, attributes: List, query_improvement: Optional[str] = None):
        self.query_improvement: Optional[str] = query_improvement
        
        self.product = product
        self.attributes = attributes
        
        self.query: str = query
        
        self.is_filtered_out: bool = False
        self.filtering_reason: str = ""
        
        self.response: str = ""

        self.evaluation: List[Tuple[str, str]] = []
        self.evaluation_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "is_filtered_out": self.is_filtered_out,
            "filtering_reason": self.filtering_reason,
            "response": self.response,
            "evaluation": self.evaluation,
            "evaluation_reason": self.evaluation_reason,
            "query_improvement": self.query_improvement,
        }


class TreeNode:
    def __init__(self,
                 value: Any,
                 parent: Optional["TreeNode"] = None,
                 children: Optional[List["TreeNode"]] = None):
        
        self.value = value
        self.parent = parent
        self.children = children if children is not None else []