from collections import deque
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from aqd.query_refinement.filter_queries import filter_queries
from aqd.query_refinement.generate_response import generate_response
from aqd.query_refinement.evaluate_responses import evaluate_responses
from aqd.schema import QueryNode, TreeNode
from aqd.query_refinement.refine_query import refine_query

QUERY_REFINEMENT_STRATEGIES = [
    """Lexical / Word-Level Strategies:
    These modify individual words but keep the overall sentence structure.
    - Synonym substitution: "cheap phones" → "affordable smartphones"
    - Morphological variation: "running shoes" → "shoes for running"
    - Spelling normalization / correction: "restuarant" → "restaurant"
    - Abbreviation expansion / compression: "AI" ↔ "artificial intelligence"
    - Lemmatization / stemming: "buying" → "buy", "bought" → "buy"
    """,

    """Syntactic Strategies:
    These change the grammatical structure or word order of a sentence.
    - Word order change: "weather in London today" → "today’s weather in London"
    - Passive ↔ Active transformation: "films directed by Nolan" ↔ "Nolan-directed films"
    - Clause restructuring / simplification: "What is the process of photosynthesis?" → "photosynthesis process"
    - Nominalization / verbalization: "how to improve learning" ↔ "improvement of learning"
    """,

    """Semantic Strategies:
    These alter meaning slightly while maintaining the original intent.
    - Generalization (hypernym substitution): "poodles" → "dogs"
    - Specialization (hyponym substitution): "dogs" → "golden retrievers"
    - Concept expansion: "climate change" → "climate change causes and effects"
    - Paraphrasing: "how to make money online" → "ways to earn income digitally"
    - Intent reframing: "buy iPhone 16" → "best deals on iPhone 16"
    """,

    """Reformulation for Context or User Intent:
    These adapt phrasing to better fit context, style, or purpose.
    - Conversational reformulation: "best Italian restaurants NYC" → "Where can I find the best Italian restaurants in NYC?"
    - Question normalization: "how old is the Eiffel Tower" → "Eiffel Tower age"
    - Perspective change: "history of AI" → "timeline of AI development"
    """
]

class QueryRefinement:
    
    def __init__(self, query_node, target_model,
                 filtering_model,
                 refinement_model, evaluation_model,
                 preferred_definition,
                 unpreferred_definition,
                 filtering_policy,
                 n_query_refiment_children):
        
        self.root: QueryNode = query_node

        self.target_model = target_model
        self.filtering_model = filtering_model
        self.refinement_model = refinement_model
        self.evaluation_model = evaluation_model

        self.preferred_definition = preferred_definition
        self.unpreferred_definition = unpreferred_definition
        self.filtering_policy = filtering_policy
        
        self.n_query_refiment_children = n_query_refiment_children

    def _process_node(self, tree_node, query_refinement_strategy_i: int):
        query_node = tree_node.value

        query_node = self.filter_queries(query_node)
        if query_node.is_filtered_out:
            tree_node.value = query_node  # keep the filtered state
            return tree_node, []

        query_node = self.generate_response(query_node)
        query_node = self.evaluate_responses(query_node)
        query_node = self.modify_responses(query_node)

        tree_node.value = query_node

        if query_refinement_strategy_i >= len(QUERY_REFINEMENT_STRATEGIES):
            return tree_node, []

        next_queries = self.refine_query(tree_node, QUERY_REFINEMENT_STRATEGIES[query_refinement_strategy_i])
        return tree_node, next_queries

    def run(self, n_workers = 5):
        """
            TAP algorithm
            https://arxiv.org/pdf/2312.02119
        """
        root = TreeNode(self.root)

        current_level = [root]
        query_refinement_strategy_i = 0

        with ThreadPoolExecutor(max_workers = n_workers) as pool:
            while current_level:
                allow_refine = query_refinement_strategy_i < len(QUERY_REFINEMENT_STRATEGIES)
                futures = [
                    pool.submit(self._process_node, node, query_refinement_strategy_i)
                    for node in current_level
                ]

                next_level = []
                for fut in as_completed(futures):
                    tree_node, next_queries = fut.result()

                    if allow_refine and next_queries:
                        for q in next_queries:
                            child = TreeNode(q, parent=tree_node)
                            tree_node.children.append(child)
                            next_level.append(child)

                query_refinement_strategy_i += 1
                current_level = next_level
        
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
    
    def refine_query(self, tree_node: TreeNode, query_refinement_strategy: str) -> List[QueryNode]:
        query_node = tree_node.value
        proposed_queries: List[Tuple[str, str]] = refine_query(tree_node,
                                                    self.refinement_model,
                                                    query_refinement_strategy,

                                                    self.n_query_refiment_children)
        next_query_nodes = []
        for query, improvement in proposed_queries:
            new_query_node = QueryNode(query, product=query_node.product, attributes=query_node.attributes, query_improvement=improvement)
            next_query_nodes.append(new_query_node)
        return next_query_nodes
    

