import argparse
from pathlib import Path
from src.config import Config





def _add_common_arguments(parser: argparse.ArgumentParser) -> None:

    parser.add_argument(
        "--task",
        type=str,
        help="task to run",
        required=True,
        choices=["ours", "tap", "pair", "gepa"],
        default="ours",
    )

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
        "--n_prompt_refinement_iterations",
        type=int,
        default=3,
        help="number of iterations of prompt optimization",
        required=False,
    )
    parser.add_argument(
        "--n_prompt_refinement_beam_search_size",
        type=int,
        default=3,
        help="beam search size in the tree of prompt optimization",
        required=False,
    )
    parser.add_argument(
        "--n_query_refinement_iterations",
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


def argument_parser() -> Config:
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
        "--task",
        type=str,
        help="task to run",
        required=True,
        choices=["ours", "tap", "pair", "gepa"],
        default="ours",
    )
    resume_parser.add_argument(
        "--prev_result",
        type=Path,
        help="path to the previous result to resume from",
        required=True,
    )
    resume_parser.add_argument(
        "--config_path",
        type=Path,
        help="path to the configuration previously used",
        required=True,
    )
    resume_parser.add_argument(
        "--output_path",
        type=Path,
        help="path to JSON to output results",
        required=True,
    )

    args = parser.parse_args()

    # Example validation: forbid prev_result on 'run' if you want
    if args.command == "run" and args.prev_result is not None:
        parser.error("--prev_result is only allowed with 'resume' or 'expand'")

    return Config(args)