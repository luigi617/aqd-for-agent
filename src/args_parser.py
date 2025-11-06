import argparse
from pathlib import Path


def argument_parser():
    """
    Required arguments:
        --input_pool: path to JSON specifying input prompt pool
        --products: path to JSON specifying products and attributes
        --filtering_policy: path to filtering policy
        --preferred_definition: path to preferred definition w/ criteria
        --unpreferred_definition: path to unpreferred definition w/ criteria
        --generate_model: path to generator (local/api) model
        --test_model: path to test (local) model
        --output_path: path to JSON to output results
    Optional arguments:
        --from_prev: path to previous results
        --n_threads: number of parallel worker threads for each LLM node (default: 1)
        --n_iterations: number of iterations of prompt optimization (default: 3)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_pool",
        type=Path,
        help="path to JSON specifying input prompt pool",
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
        "--generate_model",
        type=str,
        help="path to generator (local/api) model",
        required=True,
    )
    parser.add_argument(
        "--test_model",
        type=str,
        help="path to test (local) model",
        required=True,
    )
    parser.add_argument(
        "--from_prev",
        type=Path,
        help="path to previous results",
        default=None,
        required=False,
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
        "--n_threads",
        type=int,
        default=1,
        help="number of parallel worker threads for each LLM node",
        required=False,
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=3,
        help="number of iterations of prompt optimization",
        required=False,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="deterministic seed",
        required=False,
    )
    args = parser.parse_args()
    return args