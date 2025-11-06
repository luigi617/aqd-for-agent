import os
import requests

def get_seed():
    seed_env = os.environ.get("PYTHONHASHSEED")
    SEED = int(seed_env) if seed_env is not None else None
    return SEED

def check_ollama_running(model_path: str) -> None:
    global IS_OLLAMA_RUNNING
    """Raise error if Ollama or the requested model is not running."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if model_path not in models:
            print(f"Ollama is running but model '{model_path}' is not available.")
            raise RuntimeError(f"Ollama is running but model '{model_path}' is not available.")
    except Exception as e:
        print(f"Ollama is not running or unreachable: {e}")
        raise RuntimeError(f"Ollama is not running or unreachable: {e}")
    IS_OLLAMA_RUNNING = True