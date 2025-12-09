import json
from pathlib import Path
from src.aqd.budget import Budget

# Stretegy instructions taken from https://arxiv.org/pdf/2310.13345
DEFAULT_QUERY_REFINEMENT_STRATEGIES = [
    {
        "description": "Choose at most two words in the sentence, and change them so that they have typos.",
        "max_apply_num": 1000
    },
    {
        "description": "Change at most two letters in the sentence.",
        "max_apply_num": 1000
    },
    {
        "description": "Add at most two extraneous characters to the end of the sentence.",
        "max_apply_num": 1000
    },
    {
        "description": "Replace at most two words in the sentence with synonyms.",
        "max_apply_num": 1000
    },
    {
        "description": "Choose at most two words in the sentence that do not contribute to the meaning of the sentence and delete them.",
        "max_apply_num": 1000
    },
    {
        "description": "Add at most two semantically neutral words to the sentence.",
        "max_apply_num": 1000
    },
    {
        "description": "Add a randomly generated short meaningless handle after the sentence, such as @fasuv3.",
        "max_apply_num": 1000
    },
    {
        "description": "Paraphrase the sentence.",
        "max_apply_num": 1000
    },
    {
        "description": "Change the syntactic structure of the sentence.",
        "max_apply_num": 1000
    }
]

class Config:
    def __init__(self, args):
        self.command = args.command
        if self.command == "run":
            self.load_new_config(args)
        else:
            self.load_old_config(args)
    
    @staticmethod
    def get_strategy_info(query_refinement_strategies):
        strategies = []
        strategies_max_apply_num_count = {}
        for strategy_info in query_refinement_strategies:
            strategies.append(strategy_info["description"])
            strategies_max_apply_num_count[strategy_info["description"]] = strategy_info["max_apply_num"]
        return strategies, strategies_max_apply_num_count
    
    @staticmethod
    def get_preference_info(preferred_definition, unpreferred_definition):
        preferred_definition_res = {}
        unpreferred_definition_res = {}
        preference_score_map = {}
        for criterion, info in preferred_definition.items():
            preferred_definition_res[criterion] = info["description"]
            preference_score_map[criterion] = info["score"]

        for criterion, info in unpreferred_definition.items():
            unpreferred_definition_res[criterion] = info["description"]
            preference_score_map[criterion] = info["score"]
        return preferred_definition_res, unpreferred_definition_res, preference_score_map


    def load_new_config(self, args):
        self.seed = args.seed

        self.initial_prompt = json.load(args.init_prompt.open())
    
        self.products = json.load(args.products.open())

        self.prev_result = json.load(args.prev_result.open()) if args.prev_result else None

        self.filtering_policy = args.filtering_policy.open().read()
        
        self.original_preferred_definition = json.load(args.preferred_definition.open())
        self.original_unpreferred_definition = json.load(args.unpreferred_definition.open())
        preference_info = self.get_preference_info(
            self.original_preferred_definition,
            self.original_unpreferred_definition
        )
        self.preferred_definition = preference_info[0]
        self.unpreferred_definition = preference_info[1]
        self.preference_score_map = preference_info[2]
        
        if args.query_refinement_strategies and isinstance(args.query_refinement_strategies, Path):
            self.original_query_refinement_strategies = json.load(args.query_refinement_strategies.open())
        else:
            self.original_query_refinement_strategies = DEFAULT_QUERY_REFINEMENT_STRATEGIES
        
        strategy_info = self.get_strategy_info(self.original_query_refinement_strategies)
        self.query_refinement_strategies = strategy_info[0]
        self.strategies_max_apply_num_count = strategy_info[1]

        self.n_queries = args.n_queries
        self.n_prompt_refinemnet_iterations = args.n_prompt_refinemnet_iterations
        self.n_prompt_refinemnet_children = args.n_prompt_refinemnet_children
        self.n_prompt_refinemnet_beam_search_size = args.n_prompt_refinemnet_beam_search_size
        self.n_query_refinemnet_iterations = args.n_query_refinemnet_iterations
        self.n_query_refiment_beam_search_size = args.n_query_refiment_beam_search_size

        # divide threads into different place
        self.n_threads = args.n_threads
        base_n_threads = int(self.n_threads**(1/3))
        self.n_thread_prompt_refiment = base_n_threads
        self.n_thread_queries = self.n_threads // (base_n_threads**2)
        self.n_thread_query_refiment = base_n_threads

        self.generate_model = args.generate_model
        self.target_model = args.target_model
        self.evaluation_models = args.evaluation_models
        self.filtering_model = self.generate_model
        self.refinement_model = self.generate_model

        self.budget_limit = args.budget
        self.budget = None
        if isinstance(self.budget_limit, float) and self.budget_limit >= 0:
            self.budget = Budget(limit_usd=self.budget_limit)

        self.output_path = args.output_path


    def load_old_config(self, args):

        old_config = json.load(args.config_path.open())

        self.seed = old_config["seed"]

        self.initial_prompt = old_config["initial_prompt"]
    
        self.products = old_config["products"]

        self.prev_result = json.load(args.prev_result.open())

        self.filtering_policy = old_config["filtering_policy"]
        self.original_preferred_definition = old_config["preferred_definition"]
        self.original_unpreferred_definition = old_config["unpreferred_definition"]

        preference_info = self.get_preference_info(
            self.original_preferred_definition,
            self.original_unpreferred_definition
        )
        self.preferred_definition = preference_info[0]
        self.unpreferred_definition = preference_info[1]
        self.preference_score_map = preference_info[2]
        
        self.original_query_refinement_strategies = old_config["query_refinement_strategies"]
        strategy_info = self.get_strategy_info(self.original_query_refinement_strategies)
        self.query_refinement_strategies = strategy_info[0]
        self.strategies_max_apply_num_count = strategy_info[1]

        self.n_queries = old_config["n_queries"]
        self.n_prompt_refinemnet_iterations = old_config["n_prompt_refinemnet_iterations"]
        self.n_prompt_refinemnet_children = old_config["n_prompt_refinemnet_children"]
        self.n_prompt_refinemnet_beam_search_size = old_config["n_prompt_refinemnet_beam_search_size"]
        self.n_query_refinemnet_iterations = old_config["n_query_refinemnet_iterations"]
        self.n_query_refiment_beam_search_size = old_config["n_query_refiment_beam_search_size"]

        # divide threads into different place
        self.n_threads = old_config["n_threads"]
        self.n_thread_prompt_refiment = old_config["n_thread_prompt_refiment"]
        self.n_thread_queries = old_config["n_thread_queries"]
        self.n_thread_query_refiment = old_config["n_thread_query_refiment"]

        self.generate_model = old_config["generate_model"]
        self.target_model = old_config["target_model"]
        self.evaluation_models = old_config["evaluation_models"]
        self.filtering_model = old_config["filtering_model"]
        self.refinement_model = old_config["refinement_model"]

        self.budget_limit = old_config["budget"]
        self.budget = None
        if isinstance(self.budget_limit, float) and self.budget_limit >= 0:
            self.budget = Budget(limit_usd=self.budget_limit)

        self.output_path = args.output_path
    
    def to_dict(self):
        return {
            "seed": self.seed,
            "initial_prompt": self.initial_prompt,
            "products": self.products,
            "filtering_policy": self.filtering_policy,
            "preferred_definition": self.original_preferred_definition,
            "unpreferred_definition": self.original_unpreferred_definition,
            "query_refinement_strategies": self.original_query_refinement_strategies,
            "n_queries": self.n_queries,
            "n_prompt_refinemnet_iterations": self.n_prompt_refinemnet_iterations,
            "n_prompt_refinemnet_children": self.n_prompt_refinemnet_children,
            "n_prompt_refinemnet_beam_search_size": self.n_prompt_refinemnet_beam_search_size,
            "n_query_refinemnet_iterations": self.n_query_refinemnet_iterations,
            "n_query_refiment_beam_search_size": self.n_query_refiment_beam_search_size,
            "n_threads": self.n_threads,
            "n_thread_prompt_refiment": self.n_thread_prompt_refiment,
            "n_thread_queries": self.n_thread_queries,
            "n_thread_query_refiment": self.n_thread_query_refiment,
            "generate_model": self.generate_model,
            "target_model": self.target_model,
            "evaluation_models": self.evaluation_models,
            "filtering_model": self.filtering_model,
            "refinement_model": self.refinement_model,
            "budget": self.budget_limit,
        }
