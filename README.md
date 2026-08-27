# PQR: A Framework to Generate Diverse and Realistic User Queries that Elicit QA Agent Failures

This repo implements **PQR** (Prompt–Query-Refinement), plus the four baselines it is
compared against in the paper (GEPA, ACE, PAIR, TAP), and the evaluation pipeline.

At a high level, running an experiment always looks like:

1. **Generate** queries against the target QA agent with a method (`sh main.sh` for PQR,
   or one of the `run_*.sh` baseline scripts).
2. **Preprocess** the raw results into a common format.
3. **Evaluate** the preprocessed data to get unhelpfulness/unsafety rate, realism,
   diversity, and cost metrics.

Each step is a separate command below.

---

## 0. Setup

### 0.1 Install dependencies

```bash
pip install -e .
```

This installs its dependencies (see `pyproject.toml`), including `langchain`, `sentence-transformers`, `lexical-diversity`,
etc.

### 0.2 API keys

The generator and evaluator LLMs (`gpt-5-mini`, `gemini-2.5-flash`, `deepseek-reasoner`)
are called through LangChain, which reads credentials from environment variables.
Set whichever ones you use, e.g. in a `.env` file at the repo root (loaded automatically
via `python-dotenv`):

```bash
OPENAI_API_KEY=...
GOOGLE_API_KEY=...       # for gemini-* models
DEEPSEEK_API_KEY=...     # for deepseek-* models
```

### 0.3 Local target agent (Ollama)

The QA agent under test (`qwen3:8b` in the paper) runs locally via
[Ollama](https://docs.ollama.com/linux).

```bash
# Install (no sudo needed):
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
mkdir -p ollama && tar -C ollama -xzvf ollama-linux-amd64.tgz
export PATH=$PATH:$HOME/ollama/bin

# Start the server on a chosen port:
OLLAMA_HOST=0.0.0.0:9999 ollama serve

# Point the code at it:
export OLLAMA_URL="http://localhost:9999"

# Pull the target/generator models you plan to use, e.g.:
ollama pull qwen3:8b
ollama pull gemma3:27b
```

### 0.4 Domain knowledge / input files

All commands below take a `--products` (or `--product_file` / `--products_filepath`)
argument pointing to a JSON file of products + attributes — this is the "domain
knowledge" described in the paper (Section 3: 100 products across vacuums, diapers,
sofas, televisions). Substitute
your own file.

Sample product JSON structure:

```json
[
  {
    "product_id": "vacuum_001",
    "product_name": "SuperVac 3000",
    "category": "vacuum",
    "all_attributes": {
      "suction_power": "2000Pa",
      "battery_life": "60min",
      "weight": "3.5kg",
      "age_group": "Adult",
      ...
    },
    "selected_attributes": {
      "battery_life": "60min"
    }
  },
  {
    "product_id": "diaper_001",
    "product_name": "SoftBaby Diapers",
    "category": "diaper",
    "all_attributes": {
      "size": "M",
      "absorbency": "high",
      "material": "cotton",
      ...
    },
    "selected_attributes": {
      "size": "M"
    }
  },
  ...
]
```

The other `input_data/*.json` files are already filled in and shouldn't need editing:

| File | Purpose |
|---|---|
| `init_prompt.json` | Seed prompt for query generation (Fig. 1, "Initial prompt") |
| `preferred_definition.json` / `unpreferred_definition.json` | Objective criteria (e.g. helpfulness)|
| `filtering_policy` (`realism_definition.txt`) | Realism policy used by prompt refinement |
| `query_refinement_strategies.json` | Perturbation / role-playing strategies|

---

## 1. Generate queries

Each script below runs one method end-to-end and writes raw results (queries +
target-agent responses + reward scores) to `--output_path`.

| Method | Script | Notes |
|---|---|---|
| **PQR (ours)** | `sh main.sh` | Main method + ablations (query-only, prompt-only) + `gemma3:27b` variant, all in one file — comment out the blocks you don't need |
| GEPA | `sh run_gepa.sh` | Requires `pip install gepa` first |
| ACE | `sh run_ace.sh` | |
| PAIR | `sh run_pair.sh` | Jailbreak-style baseline |
| TAP | `sh run_tap.sh` | Jailbreak-style baseline |

Before running, open the script and:
- replace `'<your domain knowledge>'` with your products JSON path.
- adjust `--generate_model`, such as `gpt-5-mini` and `gemma3:27b`.
- adjust `--output_path` to keep runs organized (the scripts already namespace output
  by model, e.g. `output_data/gpt5/pqr/`, `output_data/gemma/pqr/`).

If a run is interrupted partway through, every script has a commented-out `resume`
command at the bottom that continues from the last saved `results_depth*.json` /
`config.json` instead of starting over — see the "Resume" note inside `main.sh` and
`run_gepa.sh`.

---

## 2. Preprocess results

Evaluation expects a common `cleaned_dataset.json` format. How you get there depends on
whether the run came from PQR/PAIR/TAP (already in the right shape) or from a
prompt-refinement baseline (GEPA/ACE), which needs one extra formatting step first.

### 2a. PQR / PAIR / TAP → straight to preprocessing

```bash
python scripts/data/data_preprocess.py \
    --input_file output_data/gpt5/pqr/results.json \
    --output_dir output_data/evaluation/gpt5/pqr \
    --filter_out_threshold 0.2 \
    --products_filepath '<your domain knowledge>' \
    --task ours   # or: pair / tap
```

### 2b. GEPA / ACE → format first, then preprocess

```bash
# Step 0: reshape the prompt-refinement log into PQR's result format
python scripts/data/baseline_data_preprocess.py \
    --input_file output_data/gpt5/gepa/gepa_prompt_log.json \
    --output_file output_data/evaluation/gpt5/gepa/formatted_results.json \
    --task gepa   # or: ace

# Step 1: same preprocessing as above, pointed at the formatted file
python scripts/data/data_preprocess.py \
    --input_file output_data/evaluation/gpt5/gepa/formatted_results.json \
    --output_dir output_data/evaluation/gpt5/gepa \
    --filter_out_threshold 0.2 \
    --products_filepath '<your domain knowledge>' \
    --task gepa
```

Both preprocessing calls write `cleaned_dataset.json` to `--output_dir`.

---

## 3. Run automatic evaluation

```bash
python -m scripts.eval.eval \
    --input_file output_data/evaluation/gpt5/pqr/cleaned_dataset.json \
    --output_dir output_data/evaluation/gpt5/pqr/eval_1_data_per_product \
    --product_file '<your domain knowledge>' \
    --embedding_model_path Qwen/Qwen3-Embedding-0.6B \
    --budget_file output_data/gpt5/pqr/budget.json \
    --task ours \
    --data_sample 1_data_per_product
```

Swap `--task`/paths for the method being evaluated (`tap`, `pair`, `gepa`, `ace`), same
as in the preprocessing step. `eval.sh` has more worked examples, including one for
scoring against real human–agent conversations (`--task real`).


### Output files

Written to `--output_dir`:

| File | Contents |
|---|---|
| `unhelpful_data.json` | All queries flagged as eliciting unhelpful/unsafe responses |
| `extracted_data.json` | The filtered subset actually used for scoring |
| `lexical_diversity.json` | `mtld_score`, `hdd`, `log_ttr`, `root_ttr`, `vocab_size`, `distincts` (Distinct@1/2/3), `ngram_entropy`, `tokens_per_turn`, `num_tokens` |
| `semantic_diversity.json` | `embedding_diversity` (1-CosSim), `n_clusters`, `cluster_diversity`, `dataset_dcscore` |
| `statistics.json` | `total_query_count`, `product_count`, `query_per_product_count`, `percentage_of_unhelpful_queries` (UHR/USR), average token cost per query |

---

## Repo map

```
src/
  main.py             entry point (`python3 -m src.main {run,resume}`)
  args_parser.py       CLI flags shared across methods
  aqd/                 PQR (our method): query + prompt refinement modules
  gepa/ ace/ pair/ tap/ baseline implementations
  models/              LLM client wrappers (OpenAI, Gemini, DeepSeek, Ollama)
scripts/
  data/                result-format conversion + preprocessing
  eval/                automatic evaluation (LLM-as-judge, realism, diversity, cost)
input_data/            domain knowledge + prompts/policies/strategies fed to all methods
```
