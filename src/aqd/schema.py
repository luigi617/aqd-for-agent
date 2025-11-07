from typing import List, Optional, Tuple


class QueryNode:
    def __init__(self, query: str, product: dict, attributes: List):
        self.query_improvement: Optional[str] = None
        
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


class QueryRefinementTreeNode:
    def __init__(self,
                 value: QueryNode,
                 parent: Optional["QueryRefinementTreeNode"] = None,
                 children: Optional[List["QueryRefinementTreeNode"]] = None):
        
        self.value = value
        self.parent = parent
        self.children = children if children is not None else []