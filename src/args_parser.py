import argparse
import json
from pathlib import Path

# Stretegy instructions taken from https://arxiv.org/pdf/2310.13345
DEFAULT_QUERY_REFINEMENT_STRATEGIES = [
    "Choose at most two words in the sentence, and change them so that they have typos.",
    "Change at most two letters in the sentence.",
    "Add at most two extraneous characters to the end of the sentence.",
    "Replace at most two words in the sentence with synonyms.",
    "Choose at most two words in the sentence that do not contribute to the meaning of the sentence and delete them.",
    "Add at most two semantically neutral words to the sentence.",
    "Add a randomly generated short meaningless handle after the sentence, such as @fasuv3.",
    "Paraphrase the sentence.",
    "Change the syntactic structure of the sentence.",
]

class Argument:
    def __init__(self, args):
        self.command = args.command

        self.seed = args.seed

        self.initial_prompt = json.load(args.init_prompt.open())
    
        self.products = json.load(args.products.open())

        self.prev_result = json.load(args.prev_result.open()) if args.prev_result else None

        self.filtering_policy = args.filtering_policy.open().read()
        self.preferred_definition = json.load(args.preferred_definition.open())
        self.unpreferred_definition = json.load(args.unpreferred_definition.open())
        
        if args.query_refinement_strategies and isinstance(args.query_refinement_strategies, Path):
            self.query_refinement_strategies = json.load(args.query_refinement_strategies.open())
        else:
            self.query_refinement_strategies = DEFAULT_QUERY_REFINEMENT_STRATEGIES

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

        self.budget = args.budget

        self.output_path = args.output_path

def _add_common_arguments(parser: argparse.ArgumentParser) -> None:

    parser.add_argument(
        "--init_prompt",
        type=Path,
        help="path to JSON specifying input prompt",
        required=True,
    )
    parser.add_argument(
        "--products",
        type=Path,
        help="path to JSON specifying products and attributes",
        required=True,
    )
    parser.add_argument(
        "--filtering_policy",
        type=Path,
        help="path to filtering policy",
        required=True,
    )
    parser.add_argument(
        "--preferred_definition",
        type=Path,
        help="path to preferred definition w/ criteria",
        required=True,
    )
    parser.add_argument(
        "--unpreferred_definition",
        type=Path,
        help="path to unpreferred definition w/ criteria",
        required=True,
    )
    parser.add_argument(
        "--query_refinement_strategies",
        type=Path,
        help="path to list of query refinement strategies",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--generate_model",
        type=str,
        help="path to generator (local/api) model",
        required=True,
    )
    parser.add_argument(
        "--evaluation_models",
        nargs="+",
        help="paths to evaluation (local/api) models",
        required=True,
    )
    parser.add_argument(
        "--target_model",
        type=str,
        help="path to test (local) model",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="path to JSON to output results",
        required=True,
    )
    parser.add_argument(
        "--n_queries",
        type=int,
        default=5,
        help="number of queries to generate per prompt template",
        required=False,
    )
    parser.add_argument(
        "--n_prompt_refinemnet_iterations",
        type=int,
        default=3,
        help="number of iterations of prompt optimization",
        required=False,
    )
    parser.add_argument(
        "--n_prompt_refinemnet_children",
        type=int,
        default=3,
        help="number of children in the tree of prompt optimization",
        required=False,
    )
    parser.add_argument(
        "--n_prompt_refinemnet_beam_search_size",
        type=int,
        default=3,
        help="beam search size in the tree of prompt optimization",
        required=False,
    )
    parser.add_argument(
        "--n_query_refinemnet_iterations",
        type=int,
        default=3,
        help="number of iterations of query refinement",
        required=False,
    )
    parser.add_argument(
        "--n_query_refiment_beam_search_size",
        type=int,
        default=3,
        help="beam search size in the tree of query refinement",
        required=False,
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        default=1,
        help="number of parallel worker threads for each LLM node",
        required=False,
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=100,
        help="budget to spent, if not set 100 usd",
        required=False,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="deterministic seed",
        required=False,
    )


def argument_parser() -> Argument:
    parser = argparse.ArgumentParser(prog="pipeline")

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="which mode to run: run / resume / expand",
    )

    # ---- run ----
    run_parser = subparsers.add_parser(
        "run", help="fresh run from scratch"
    )
    _add_common_arguments(run_parser)
    # For 'run', prev_result is optional (or could be omitted entirely)
    run_parser.add_argument(
        "--prev_result",
        type=Path,
        help="path to the previous result (usually not used in 'run')",
        default=None,
        required=False,
    )

    # ---- resume ----
    resume_parser = subparsers.add_parser(
        "resume", help="resume from previous result"
    )
    resume_parser.add_argument(
        "--prev_result",
        type=Path,
        help="path to the previous result to resume from",
        required=True,
    )

    # ---- expand ----
    expand_parser = subparsers.add_parser(
        "expand", help="expand using previous result but allow different config"
    )
    expand_parser.add_argument(
        "--prev_result",
        type=Path,
        help="path to the previous result to expand from",
        required=True,
    )
    expand_parser.add_argument(
        "--n_prompt_refinemnet_iterations",
        type=int,
        default=3,
        help="new number of iterations of prompt optimization",
        required=False,
    )

    args = parser.parse_args()

    # Example validation: forbid prev_result on 'run' if you want
    if args.command == "run" and args.prev_result is not None:
        parser.error("--prev_result is only allowed with 'resume' or 'expand'")

    return Argument(args)