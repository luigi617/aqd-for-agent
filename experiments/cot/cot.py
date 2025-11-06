#!/usr/bin/env python3
import sys

import json
import time
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any, Literal, Type

from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from scipy import stats
from pydantic import BaseModel, ValidationError, Field

from openai import OpenAI

sys.path.append(str(Path(__file__).parent.parent))

from pipeline.misc.model import get_basic_model

EVAL_STRING_LLM_PYDANTIC_TEMPLATE = """\
    You are a product question answering task evaluator.
    You will be given a product question and a model-generated answer. 
    Please do not generate any opening, closing, and explanations.
    The response should be one word with a number between 0 and 3 only.

    Evaluation Criteria (0–3):
    0: The answer was irrelevant to the question.
    1: The answer was related but did not answer the question.
    2: The answer partially answered the question.
    3: The answer directly answered the question.

    Example:
    Product Question:
    What material is the phone case made of?

    Model Generated Answer:
    It’s made from flexible TPU plastic.

    Final Score (only the number on this line):
    3

    Now evaluate the following:
    Product Question:
    {query}

    Model Generated Answer:
    {response}
"""

EVAL_STRING_LLM_PYDANTIC_COT_TEMPLATE = """\
    You are a product question answering task evaluator.
    You will be given a product question and a model-generated answer. 
    First, think step by step about how well the answer addresses the question according to the evaluation criteria.
    Then, at the end, output only the final numeric score (0–3) on its own line.

    Evaluation Criteria (0–3):
    0: The answer was irrelevant to the question.
    1: The answer was related but did not answer the question.
    2: The answer partially answered the question.
    3: The answer directly answered the question.

    Example:
    Product Question:
    Is this difficult to clean?

    Model Generated Answer:
    it can be a little tedious to clean.

    Your reasoning:
    1. Understanding the Product Question
    The question asks, "Is this difficult to clean?" The user is seeking information about the ease or difficulty of cleaning the product. The expected answer should clearly address whether the product is easy or difficult to clean.
    2. Analyzing the model-generated Answer
    The model-generated answer says, "it can be a little tedious to clean."
    The model-generated answer indicates that cleaning the product may not be easy and can take some effort, which aligns with the user’s inquiry about difficulty.
    The phrase ‘a little tedious’ implies some difficulty, though not extreme, and provides a direct answer to the question.
    3. Rating the model-generated Answer Based on the Evaluation Criteria
    Score 3: The model-generated answer directly answers the question by explaining that cleaning the product is somewhat difficult or tedious. It addresses the key aspect of the user’s question.
    4. Conclusion
    The model-generated answer is relevant, clear, and directly addresses the question about difficulty in cleaning.

    Final Score (only the number on this line):
    3

    Now evaluate the following:
    Product Question:
    {query}

    Model Generated Answer:
    {response}
"""

EVAL_STRING_LLM_COT_TEMPLATE = """\
    You are a product question answering task evaluator.
    You will be given a product question and a model-generated answer. 
    First, think step by step about how well the answer addresses the question according to the evaluation criteria.
    Then, at the end, output only the final numeric score (0–3) on its own line.

    Evaluation Criteria (0–3):
    0: The answer was irrelevant to the question.
    1: The answer was related but did not answer the question.
    2: The answer partially answered the question.
    3: The answer directly answered the question.

    Example:
    Product Question:
    Is this difficult to clean?

    Model Generated Answer:
    it can be a little tedious to clean.

    Your reasoning:
    1. Understanding the Product Question
    The question asks, "Is this difficult to clean?" The user is seeking information about the ease or difficulty of cleaning the product. The expected answer should clearly address whether the product is easy or difficult to clean.
    2. Analyzing the model-generated Answer
    The model-generated answer says, "it can be a little tedious to clean."
    The model-generated answer indicates that cleaning the product may not be easy and can take some effort, which aligns with the user’s inquiry about difficulty.
    The phrase ‘a little tedious’ implies some difficulty, though not extreme, and provides a direct answer to the question.
    3. Rating the model-generated Answer Based on the Evaluation Criteria
    Score 3: The model-generated answer directly answers the question by explaining that cleaning the product is somewhat difficult or tedious. It addresses the key aspect of the user’s question.
    4. Conclusion
    The model-generated answer is relevant, clear, and directly addresses the question about difficulty in cleaning.

    Final Score (only the number on this line):
    3

    Now evaluate the following:
    Product Question:
    {query}

    Model Generated Answer:
    {response}

    Your reasoning:
"""

# ---------- CONFIG ----------
BENCHMARK_DIR = Path("benchmark_data/test")


# Pydantic Schemas
class EvalNoCoT(BaseModel):
    score: int = Field(ge=0, le=3)

class EvalWithCoT(BaseModel):
    score: int = Field(ge=0, le=3)
    reasoning: Optional[str] = None

# Retry/backoff
MAX_RETRIES = 5
BACKOFF_SECONDS = 2.0


def model_with_pydantic(prompt: str, model: str, pydantic_model: Type[BaseModel]) -> str:
    
    model_chain = get_basic_model(
        model_path=model,
        pydantic_model=pydantic_model,
        temperature=0.0,
    )
    return model_chain(prompt)

def model_with_completion(prompt: str, model: str) -> str:
    client = OpenAI()
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a product question answering task evaluator."},
                    {"role": "user", "content": prompt},
                ],
            )
            content = resp.choices[0].message.content
            if not content:
                raise ValueError("Empty completion content")
            return content.strip()
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(BACKOFF_SECONDS * (attempt + 1))
    raise RuntimeError("unreachable")

def parse_int_last_char(text: str) -> int:
    """
    Extract the last digit 0–5 from a string (as per user’s original logic).
    """
    digits = [ch for ch in text if ch.isdigit()]
    if not digits:
        raise ValueError(f"No digits found in output: {text!r}")
    last = int(digits[-1])
    if last < 0 or last > 5:
        raise ValueError(f"Digit out of range 0–5: {last} (from {text!r})")
    return last

# Method 1: Pydantic without CoT (schema-only)
def run_pydantic_no_cot(prompt: str, model: str) -> int:
    result = model_with_pydantic(prompt, model, EvalNoCoT)
    return (result["score"], )

# Method 2: Pydantic with CoT (schema + reasoning)
def run_pydantic_with_cot(prompt: str, model: str) -> int:
    result = model_with_pydantic(prompt, model, EvalWithCoT)
    return result["score"], result["reasoning"]

# Method 3: No Pydantic, with CoT — plain completion and "last-digit" parse
def run_completion_with_cot(prompt: str, model: str) -> int:
    out = model_with_completion(prompt, model)
    return parse_int_last_char(out), out


# ---------- Runner ----------
def run_one_file(
    csv_path: Path,
    model_name: str,
    method_label: str,
) -> Tuple[float, float]:
    """
    Returns (spearman, pearson) for one file, one model+method combo.
    """
    # Read TSV (tab delimiter)
    df = pd.read_csv(csv_path, delimiter='	')
    results: List[int] = []
    
    run_fn = None
    
    if method_label == "pydantic_no_cot":
        run_fn = lambda query, response: run_pydantic_no_cot(
            EVAL_STRING_LLM_PYDANTIC_TEMPLATE.format(query=query, response=response),
            model_name,
        )
    elif method_label == "pydantic_with_cot":
        run_fn = lambda query, response: run_pydantic_with_cot(
            EVAL_STRING_LLM_PYDANTIC_COT_TEMPLATE.format(query=query, response=response),
            model_name,
        )
    elif method_label == "completion_with_cot":
        run_fn = lambda query, response: run_completion_with_cot(
            EVAL_STRING_LLM_COT_TEMPLATE.format(query=query, response=response),
            model_name,
        )
    else:
        raise ValueError(f"Unknown method_label: {method_label}")

    results = []
    all_info_res = []
    for _, row in tqdm(df.iterrows(), desc=f"{csv_path.stem} [{model_name}/{method_label}]"):
        res = run_fn(row["question"], row["answer"])
        results.append(int(res[0]))
        all_info_res.append(res)

    results_truth = df["annotated"].astype(int).to_list()
    sp_coeff, _ = stats.spearmanr(results, results_truth)
    pe_coeff, _ = stats.pearsonr(results, results_truth)
    return float(sp_coeff), float(pe_coeff), all_info_res, results_truth

def main():
    parser = argparse.ArgumentParser(description="Run human-eval comparisons.")
    parser.add_argument(
        "--output_path",
        default="comparison_results.csv",
        help="Path to save the comparison table as CSV.",
    )
    # gpt-4o-mini, gpt-5-nano, gpt-5-mini
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Which model families to run (keys of MODEL_IDS).",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["pydantic_no_cot", "pydantic_with_cot", "completion_with_cot"],
        help="Which methods to run.",
    )
    args = parser.parse_args()

    rows = []
    files = sorted(
        [p for p in BENCHMARK_DIR.iterdir() if p.is_file() and p.suffix == ".csv"]
    )
    if not files:
        print(f"No CSV files found in {BENCHMARK_DIR} (tab-delimited expected).")
        return

    # Run experiments
    for csv_path in files:
        for method_label in args.methods:
            sp, pe, res, truth_scores = run_one_file(
                csv_path=csv_path,
                model_name=args.model,
                method_label=method_label,
            )
            print(f"{csv_path.stem} | {args.model} | {method_label} | Spearman={sp:.4f} | Pearson={pe:.4f}")
            rows.append(
                {
                    "file": csv_path.stem,
                    "model": args.model,
                    "method": method_label,
                    "spearman": sp,
                    "pearson": pe,
                    "truth_scores": truth_scores,
                    "results": res,
                }
            )
    with open(args.output_path, 'w') as f:
        json.dump(rows, f, indent=4)

if __name__ == "__main__":
    load_dotenv(override=True)
    main()