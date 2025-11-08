from collections import deque
from tqdm import tqdm
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from aqd.query_refinement.filter_queries import filter_queries
from aqd.query_refinement.generate_response import generate_response
from aqd.query_refinement.evaluate_responses import evaluate_responses
from aqd.schema import QueryNode, TreeNode
from aqd.query_refinement.refine_query import refine_query

QUERY_REFINEMENT_STRATEGIES = [
    """Lexical / Word-Level Strategies:
    Modify or normalize individual words or short phrases without changing grammatical structure or meaning scope.
    - Domain-Specific Term Replacement: Shifts the model into a different domain register, e.g., "heart attack" -> "myocardial infarction"; "AI tools" -> "machine learning frameworks".
    - Sentiment Shift / Synonym Substitution: Replaces words with synonyms or adjusts emotional orientation, possibly shifting nuance, e.g., "cheap phones" -> "affordable smartphones".
    - Negation / Polarity Insertion: Reverses propositional truth with a single lexical token, e.g., "smoking causes cancer" -> "smoking does not cause cancer".
    """,

    """Syntactic Strategies:
    Change the grammatical or structural form of a sentence while preserving meaning.
    - Coordination / Subordination Swap: Reorganizes clause relationships (parallel vs. causal), e.g., "Photosynthesis uses sunlight and produces oxygen" ↔ "Oxygen is produced because photosynthesis uses sunlight".
    - Clause Restructuring / Simplification: Removes or flattens subordinate clauses or question forms, e.g., "What is the process of photosynthesis?" -> "photosynthesis process".
    - Question ↔ Statement Transformation: Converts interrogative to declarative form (and vice versa), e.g., "Where is Mount Everest located?" ↔ "Mount Everest location".
    """,

    """Semantic Strategies:
    Modify the meaning or conceptual scope of the query while keeping user intent recognizable.
    - Generalization: Replaces a specific term with a broader category (moves up the conceptual hierarchy), e.g., "poodles" -> "dogs".
    - Specialization: Narrows the concept to a more specific instance or subtype, e.g., "dogs" -> "golden retrievers".
    - Concept Expansion / Specification: Adds semantically related facets or dimensions to enrich the topic, e.g., "climate change" -> "climate change causes and effects".
    - Intent Reframing for Purpose: Alters the user's intent type, e.g., "buy iPhone 16" -> "best deals on iPhone 16".
    - Attribute or Property Focus Shift: Zooms in on a specific attribute of the concept, e.g., "electric cars" -> "electric car battery life".
    """,

    """Contextual / Intent-Aware Reformulation:
    Adjust phrasing to suit conversational, stylistic, or situational context without altering factual meaning.
    - Conversational Adaptation: Converts keyword-style input into a natural conversational query, e.g., "best Italian restaurants NYC" -> "Where can I find the best Italian restaurants in NYC?".
    - Perspective or Role Shift: Changes narrative perspective (actor, agent, or focus role), e.g., "how did humans develop AI?" -> "how did AI development shape human society?".
    - Task Type Conversion: Switches the task framing (definition -> summary -> instruction -> analysis), e.g., "how to fix a printer" -> "step-by-step guide to fixing a printer".
    - Audience Adaptation: Adjusts difficulty, tone, and style for a specific audience or role, e.g., "quantum entanglement explained" -> "explain quantum entanglement to a 10-year-old".
    - Discourse Function Adjustment: Changes the communicative purpose (listing -> reasoning -> justification), e.g., "examples of renewable energy" -> "why are renewable energies important?".
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
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Query refinement: depth {query_refinement_strategy_i}", leave=False):
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
    

