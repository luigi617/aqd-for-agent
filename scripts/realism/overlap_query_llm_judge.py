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

# ====== NEW INPUT/OUTPUT ======
INPUT_JSON = "unhelpful_data/results/queries_in_jsons_with_realism.json"
OUTPUT_JSON = "unhelpful_data/results/queries_with_llm_realism.json"
OUTPUT_DIR = "unhelpful_data/results/"

DEV = True
DEV_MAX_ITEMS = None  # cap total items in dev mode (set None for no cap)

OPENAI_MODEL = "gpt-5-nano" if DEV else "gpt-5.2"
DEEPSEEK_MODEL = "deepseek-chat" if DEV else "deepseek-reasoner"
GEMINI_MODEL = "gemini-2.0-flash-lite" if DEV else "gemini-2.5-flash"

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

MAX_CONCURRENT_REQUESTS = 30
MAX_CONCURRENT_ITEMS = 200  # overall queue workers

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


def _extract_first_json_object(text: str) -> Optional[str]:
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
    try:
        return json.loads(text)
    except Exception:
        chunk = _extract_first_json_object(text)
        if chunk:
            return json.loads(chunk)
        raise


async def atomic_write_json(path: str, obj: Any) -> None:
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

    return {
        "avg_score": avg,
        "scores": scores,
        "errors": errors,
        "per_model": results,
    }


async def main():
    ensure_dirs()

    # ---- read input list ----
    async with aiofiles.open(INPUT_JSON, "r", encoding="utf-8") as f:
        raw = await f.read()

    data = json.loads(raw)
    if not isinstance(data, list):
        raise SystemExit("INPUT_JSON must be a list of dicts like {query, realism}")

    # validate + optional dev cap
    items: List[Dict[str, Any]] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        q = obj.get("query")
        r = obj.get("realism")
        if isinstance(q, str) and q.strip():
            items.append({"query": q, "realism": r})
        if DEV and DEV_MAX_ITEMS is not None and len(items) >= DEV_MAX_ITEMS:
            break

    if not items:
        raise SystemExit("No valid items found in input JSON")

    # ---- async queue of queries ----
    q: asyncio.Queue = asyncio.Queue()
    for it in items:
        await q.put(it)

    results: List[Dict[str, Any]] = []
    results_lock = asyncio.Lock()

    out_path = OUTPUT_JSON
    checkpoint_every = 20
    processed = 0
    processed_lock = asyncio.Lock()

    async with aiohttp.ClientSession() as session:

        async def worker():
            nonlocal processed
            while True:
                it = await q.get()
                try:
                    query = it["query"]
                    realism = it.get("realism")

                    polled = await poll_judge(session, query)

                    row = {
                        "query": query,
                        "realism": realism,
                        "llm_realism": polled["avg_score"],  # <--- your requested field
                        "per_model_scores": polled["scores"],
                        "per_model_errors": polled["errors"],
                    }

                    async with results_lock:
                        results.append(row)

                    do_ckpt = False
                    async with processed_lock:
                        processed += 1
                        if processed % checkpoint_every == 0:
                            do_ckpt = True

                    if do_ckpt:
                        async with results_lock:
                            snapshot = list(results)
                        await atomic_write_json(out_path, snapshot)

                finally:
                    q.task_done()

        num_workers = min(MAX_CONCURRENT_ITEMS, max(1, len(items)))
        workers = [asyncio.create_task(worker()) for _ in range(num_workers)]

        await q.join()

        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    # final write
    async with results_lock:
        final_snapshot = list(results)
    await atomic_write_json(out_path, final_snapshot)

    print(f"Done. Wrote: {out_path} ({len(final_snapshot)} items)")


if __name__ == "__main__":
    asyncio.run(main())
