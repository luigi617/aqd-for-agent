import os
import json
import glob
import time
import asyncio
from typing import Any, Dict, List, Optional, Callable
from dotenv import load_dotenv
import aiohttp
import aiofiles
from openai import AsyncOpenAI
load_dotenv()

INPUT_DIR = "unhelpful_data"
OUTPUT_DIR = "unhelpful_data/results"

DEV = True
DEV_ITEMS_PER_FILE = None

OPENAI_MODEL = "gpt-5-nano" if DEV else "gpt-5.2"
DEEPSEEK_MODEL = "deepseek-chat" if DEV else "deepseek-reasoner"
GEMINI_MODEL = "gemini-2.0-flash-lite" if DEV else "gemini-2.5-flash"

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


MAX_CONCURRENT_REQUESTS = 30
MAX_CONCURRENT_FILES = 8

MAX_CONCURRENT_ITEMS_PER_FILE = 50


MAX_CONCURRENT_GPT = 10
MAX_CONCURRENT_DEEPSEEK = 10
MAX_CONCURRENT_GEMINI = 10

MAX_RETRIES = 3
RETRY_BASE_SECONDS = 0.8

HTTP_TIMEOUT_S = 90

PROMPT_TEMPLATE = """You are an expert e-commerce query evaluator. Given a single user query to a shopping assistant, judge how REALISTIC / HUMAN-LIKE it is (i.e., how likely it was written by a real shopper). Assign a score from 1 to 5 and return JSON only. Return the score and reasoning in JSON FORMAT.

Format:
{{
  "reason": "<1-3 concise sentences citing the key cues>",
  "score": <integer 1-5>
}}

HUMAN-LIKENESS (naturalness as a shopper query)
Description:

Evaluation Criteria:

5 = Clearly realistic (real shopper).
- Feels authentic and spontaneous.
- Prioritizes efficiency over formal correctness.
- Looks like what real shoppers type: short keywords ("durable running shoes") or a brief natural question ("best blender for smoothies?").
- No unnecessary backstory, no rigid structure, no machine artifacts.

4 = Mostly realistic (plausible shopper).
- Still looks like a genuine shopper query, but with minor awkwardness (slightly verbose, slightly unnatural phrasing).
- May include a little extra context, but not a narrative/monologue.
- Not overly technical, not overly formal, not repetitive.

3 = Borderline realism (could be real, could be synthetic).
- Some signs of templating or unnaturalness: odd phrasing, mildly rigid formatting(i.e includes underscore), or mild repetition.
- Still could plausibly be typed by a real user, but it doesn't feel fully natural.

2 = Likely unrealistic (synthetic / machine-like).
- Reads more like a computer command, structured data entry, or a template than a shopper query.
- Often overly formal or list-like (e.g., "(blue shirt, large, cotton)"), contains unnecessary narration, or repeats information unnaturally.
- May feel assistant-like ("I would like to request…") rather than shopper-like.

1 = Clearly unrealistic (not a real shopper query).
- Clearly machine-generated or nonsensical.
- Strong machine artifacts: SQL/code ("SELECT * FROM…"), prompt text, SKU-heavy/jargon-only expert strings, or long narrative paragraphs with backstory.
- Not something a typical shopper would type to a shopping assistant.


QUERY: {query}


Evaluation Form:"""


def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def clamp_score(x: Any) -> Optional[int]:
    try:
        v = int(x)
    except Exception:
        return None
    return max(1, min(5, v))

def make_key(file_name: str, item: Dict[str, Any], query: str) -> str:
    gid = item.get("global_id", None)
    if gid is None:
        gid = item.get("id", None)
    if gid is None:
        gid = f"row{item.get('_index', 'na')}"
    return f"{file_name}_{gid}_{query}"

def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Best-effort: pull the first {...} block from text.
    Helps when models accidentally wrap JSON in prose or code fences.
    """
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None

def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Strict first, then salvage first JSON object if needed.
    Raises if cannot parse anything JSON-like.
    """
    try:
        return json.loads(text)
    except Exception:
        chunk = _extract_first_json_object(text)
        if chunk:
            return json.loads(chunk)
        raise

async def atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    tmp_path = path + ".tmp"
    async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(obj, ensure_ascii=False, indent=2))
    os.replace(tmp_path, path)


async def call_with_retries(coro_factory: Callable[[], Any]):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return await coro_factory()
        except Exception as e:
            last_err = e
            sleep_s = (RETRY_BASE_SECONDS * (2 ** attempt)) + (0.1 * attempt)
            await asyncio.sleep(sleep_s)
    raise last_err


request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
sem_gpt = asyncio.Semaphore(MAX_CONCURRENT_GPT)
sem_deepseek = asyncio.Semaphore(MAX_CONCURRENT_DEEPSEEK)
sem_gemini = asyncio.Semaphore(MAX_CONCURRENT_GEMINI)

_openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def judge_with_gpt(query: str) -> Dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(query=query)

    async with request_semaphore:
        async with sem_gpt:
            resp = await call_with_retries(
                lambda: _openai_client.responses.create(
                    model=OPENAI_MODEL,
                    input=prompt,
                    text={"format": {"type": "json_object"}},
                )
            )

    out_text = resp.output_text or ""
    data = safe_json_loads(out_text)

    return {
        "provider": "gpt",
        "model": OPENAI_MODEL,
        "raw": data,
        "score": clamp_score(data.get("score")),
        "reason": str(data.get("reason", "")).strip(),
    }

async def _post_json(
    session: aiohttp.ClientSession,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S)
    async with session.post(url, headers=headers, json=payload, timeout=timeout) as r:
        txt = await r.text()
        if r.status >= 400:
            raise RuntimeError(f"HTTP {r.status} from {url}: {txt[:500]}")
        return json.loads(txt)

# DeepSeek (OpenAI-compatible)
async def judge_with_deepseek(session: aiohttp.ClientSession, query: str) -> Dict[str, Any]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    prompt = PROMPT_TEMPLATE.format(query=query)
    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    async with request_semaphore:
        async with sem_deepseek:
            resp_json = await call_with_retries(lambda: _post_json(session, url, headers, payload))

    content = resp_json["choices"][0]["message"]["content"]
    data = safe_json_loads(content)

    return {
        "provider": "deepseek",
        "model": DEEPSEEK_MODEL,
        "raw": data,
        "score": clamp_score(data.get("score")),
        "reason": str(data.get("reason", "")).strip(),
    }

# Gemini (Google Generative Language REST)
async def judge_with_gemini(session: aiohttp.ClientSession, query: str) -> Dict[str, Any]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")

    prompt = PROMPT_TEMPLATE.format(query=query)
    url = GEMINI_ENDPOINT.format(model=GEMINI_MODEL)
    url = f"{url}?key={api_key}"

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            # "maxOutputTokens": 256,
            "responseMimeType": "application/json",
        },
    }

    async with request_semaphore:
        async with sem_gemini:
            resp_json = await call_with_retries(
                lambda: _post_json(session, url, headers={"Content-Type": "application/json"}, payload=payload)
            )

    text = (
        resp_json.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
    )
    data = safe_json_loads(text)

    return {
        "provider": "gemini",
        "model": GEMINI_MODEL,
        "raw": data,
        "score": clamp_score(data.get("score")),
        "reason": str(data.get("reason", "")).strip(),
    }


async def poll_judge(session: aiohttp.ClientSession, query: str) -> Dict[str, Any]:
    async def run_one(name: str):
        if name == "gpt":
            return await judge_with_gpt(query)
        if name == "deepseek":
            return await judge_with_deepseek(session, query)
        if name == "gemini":
            return await judge_with_gemini(session, query)
        raise ValueError("unknown model name")

    tasks = {
        "gpt": asyncio.create_task(run_one("gpt")),
        "deepseek": asyncio.create_task(run_one("deepseek")),
        "gemini": asyncio.create_task(run_one("gemini")),
    }

    results: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    for name, t in tasks.items():
        try:
            results[name] = await t
        except Exception as e:
            errors[name] = f"{type(e).__name__}: {e}"

    scores = {k: (v.get("score") if isinstance(v, dict) else None) for k, v in results.items()}
    numeric_scores = [s for s in scores.values() if isinstance(s, int)]
    avg = (sum(numeric_scores) / len(numeric_scores)) if numeric_scores else None

    reasons = {k: (v.get("reason") if isinstance(v, dict) else None) for k, v in results.items()}

    return {
        "avg_score": avg,
        "scores": scores,
        "reasons": reasons,
        "errors": errors,
        "per_model": results,
    }


async def process_file(
        session: aiohttp.ClientSession,
        path: str,
        on_row: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    ) -> Dict[str, Any]:
    file_name = os.path.basename(path)

    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        raw = await f.read()

    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("Top-level JSON must be a list of dicts.")
    except Exception as e:
        out = {
            "file": file_name,
            "ok": False,
            "error": f"Failed to parse input JSON: {e}",
            "results": {},
        }
        await _write_out(file_name, out)
        return out

    for i, item in enumerate(data):
        if isinstance(item, dict):
            item["_index"] = i

    results_dict: Dict[str, Any] = {}
    results_lock = asyncio.Lock()

    q: asyncio.Queue = asyncio.Queue()

    # Enqueue valid items
    num_enqueued = 0
    cap = DEV_ITEMS_PER_FILE

    for item in data:
        if cap is not None and num_enqueued >= cap:
            break

        if not isinstance(item, dict):
            continue
        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            continue
        await q.put(item)
        num_enqueued += 1

    async def worker():
        while True:
            item = await q.get()
            try:
                query = item["query"]
                key = make_key(file_name, item, query)
                polled = await poll_judge(session, query)

                row = {
                    "file": file_name,
                    "global_id": item.get("global_id"),
                    "id": item.get("id"),
                    "query": query,
                    "avg_score": polled["avg_score"],
                    "scores": polled["scores"],
                    "reasons": polled["reasons"],
                    "errors": polled["errors"],
                    "per_model": polled["per_model"],
                }
                async with results_lock:
                    results_dict[key] = row
                if on_row is not None:
                    await on_row(key, row)
            finally:
                q.task_done()

    # Start bounded workers for this file
    num_workers = min(MAX_CONCURRENT_ITEMS_PER_FILE, max(1, num_enqueued))
    workers = [asyncio.create_task(worker()) for _ in range(num_workers)]

    # Wait for all items to be processed
    await q.join()

    # Cancel workers cleanly
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

    out = {
        "file": file_name,
        "ok": True,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_items": len(data),
        "num_enqueued": num_enqueued,
        "num_judged": len(results_dict),
        "results": results_dict,
    }

    await _write_out(file_name, out)
    return out

async def _write_out(file_name: str, obj: Dict[str, Any]) -> None:
    out_path = os.path.join(OUTPUT_DIR, file_name)
    async with aiofiles.open(out_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(obj, ensure_ascii=False, indent=2))

async def main():
    ensure_dirs()
    paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    if not paths:
        raise SystemExit(f"No JSON files found in: {INPUT_DIR}")

    file_semaphore = asyncio.Semaphore(MAX_CONCURRENT_FILES)

    # Shared combined state (across all files)
    combined_results: Dict[str, Any] = {}
    combined_lock = asyncio.Lock()
    combined_write_lock = asyncio.Lock()
    judged_counter = 0  # global judged rows
    counter_lock = asyncio.Lock()

    # Track per-file progress for nicer checkpoint metadata
    files_progress: Dict[str, Dict[str, Any]] = {
        os.path.basename(p): {"file": os.path.basename(p), "ok": None, "num_judged": 0, "error": None}
        for p in paths
    }
    files_progress_lock = asyncio.Lock()

    combined_path = os.path.join(OUTPUT_DIR, "_combined.json")

    async def write_combined_checkpoint() -> None:
        # Only one writer at a time
        async with combined_write_lock:
            async with combined_lock:
                snapshot_results = dict(combined_results)
            async with files_progress_lock:
                files_summary = [dict(v) for v in files_progress.values()]

            combined_out = {
                "input_dir": INPUT_DIR,
                "output_dir": OUTPUT_DIR,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "providers": {
                    "gpt": {"model": OPENAI_MODEL, "max_concurrent": MAX_CONCURRENT_GPT},
                    "deepseek": {
                        "model": DEEPSEEK_MODEL,
                        "base_url": DEEPSEEK_BASE_URL,
                        "max_concurrent": MAX_CONCURRENT_DEEPSEEK,
                    },
                    "gemini": {"model": GEMINI_MODEL, "max_concurrent": MAX_CONCURRENT_GEMINI},
                },
                "limits": {
                    "max_concurrent_requests_total": MAX_CONCURRENT_REQUESTS,
                    "max_concurrent_files": MAX_CONCURRENT_FILES,
                    "max_concurrent_items_per_file": MAX_CONCURRENT_ITEMS_PER_FILE,
                    "max_retries": MAX_RETRIES,
                },
                "files": files_summary,
                "results": snapshot_results,
            }

            await atomic_write_json(combined_path, combined_out)

    async with aiohttp.ClientSession() as session:

        async def on_row(key: str, row: Dict[str, Any]) -> None:
            nonlocal judged_counter

            # Update shared combined dict
            async with combined_lock:
                combined_results[key] = row

            # Update per-file progress
            file_name = row.get("file")
            if file_name:
                async with files_progress_lock:
                    fp = files_progress.get(file_name)
                    if fp is not None:
                        fp["num_judged"] = int(fp.get("num_judged", 0)) + 1

            # Every 10 results, checkpoint _combined.json
            do_write = False
            async with counter_lock:
                judged_counter += 1
                if judged_counter % 10 == 0:
                    do_write = True

            if do_write:
                await write_combined_checkpoint()

        async def process_limited(p: str):
            async with file_semaphore:
                file_name = os.path.basename(p)
                try:
                    out = await process_file(session, p, on_row=on_row)
                except Exception as e:
                    out = {"file": file_name, "ok": False, "error": str(e), "results": {}}

                # Mark file completion status
                async with files_progress_lock:
                    fp = files_progress.get(file_name)
                    if fp is not None:
                        fp["ok"] = bool(out.get("ok"))
                        fp["error"] = out.get("error")

                return out

        per_file = await asyncio.gather(*(process_limited(p) for p in paths))

    # Final combined write (ensures last partial <10 also flushed)
    await write_combined_checkpoint()

    print(f"Done. Wrote per-file outputs to: {OUTPUT_DIR}")
    print(f"Combined results (checkpointed): {combined_path}")

if __name__ == "__main__":
    asyncio.run(main())
