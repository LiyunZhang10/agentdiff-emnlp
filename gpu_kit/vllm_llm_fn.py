"""
gpu_kit/vllm_llm_fn.py

A drop-in `llm_fn(prompt) -> str` that talks to a local vllm OpenAI-compatible
server. Reuses the rest of agentdiff_v2.py untouched.

Usage:
    from vllm_llm_fn import make_llm_fn
    llm_fn = make_llm_fn(model="Qwen/Qwen2.5-14B-Instruct",
                         base_url="http://localhost:8000/v1",
                         temperature=0.0)
    out = llm_fn("What is 2+2?")
"""
import os
import time
import json
import requests


def make_llm_fn(model, base_url="http://localhost:8000/v1", api_key="EMPTY",
                temperature=0.0, max_tokens=1024, timeout=180, max_retries=4):
    """Return a closure llm_fn(prompt: str) -> str that hits vllm /v1/chat/completions."""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    def llm_fn(prompt):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        last_err = None
        for attempt in range(max_retries):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=timeout)
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                    time.sleep(2 ** attempt)
                    continue
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = repr(e)
                time.sleep(2 ** attempt)
        # graceful degrade — pipeline tolerates empty strings
        return f"[LLM_ERROR] {last_err}"

    return llm_fn


if __name__ == "__main__":
    # smoke test
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-14B-Instruct"
    base = os.environ.get("VLLM_BASE", "http://localhost:8000/v1")
    fn = make_llm_fn(model=model, base_url=base)
    print(fn("Reply with exactly: PONG"))
