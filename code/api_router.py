# -*- coding: utf-8 -*-
"""
AgentDiff 实验 - API 路由器
支持 Ollama (本地) + Groq + Gemini 免费 API
兼容 Python 3.6+
"""
import os
import json
import time
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def ollama_call(prompt, model="qwen2.5:3b", temperature=0.0):
    """调用本地 Ollama 模型"""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # keep_alive=30m: 让目标模型粘在内存里，避免反复 load/unload 抢核
        "keep_alive": "30m",
        "options": {
            "temperature": temperature,
            "num_ctx": 4096,
            # 限制单次最大输出 token，避免长尾推理无限延伸（agent 答案不需要太长）
            "num_predict": 1024,
        }
    }
    # NOTE: under heavy CPU load (load avg ~48 on a 48-core box), a single
    # generation on Llama-8B can take 5-10 minutes. Give 60min hard ceiling
    # per HTTP call, and only retry once to avoid request pile-up.
    for attempt in range(2):
        try:
            resp = requests.post(url, json=payload, timeout=3600)
            resp.raise_for_status()
            return resp.json()["response"]
        except Exception as e:
            logger.warning("Ollama attempt %d failed: %s", attempt + 1, str(e))
            time.sleep(5)
    raise RuntimeError("Ollama failed after 2 attempts")


def groq_call(prompt, model="llama-3.3-70b-versatile", temperature=0.0):
    """调用 Groq 免费 API"""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    for attempt in range(5):
        try:
            time.sleep(2)  # 30 RPM -> 2s min
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            wait = min(60, 4 * (2 ** attempt))
            logger.warning("Groq attempt %d failed: %s, waiting %ds", attempt + 1, str(e), wait)
            time.sleep(wait)
    raise RuntimeError("Groq failed after 5 attempts")


def gemini_call(prompt, temperature=0.0):
    """调用 Google AI Studio Gemini 免费 API"""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature}
    }
    for attempt in range(5):
        try:
            time.sleep(4)  # 15 RPM -> 4s min
            resp = requests.post(url + "?key=" + api_key, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            wait = min(60, 4 * (2 ** attempt))
            logger.warning("Gemini attempt %d failed: %s, waiting %ds", attempt + 1, str(e), wait)
            time.sleep(wait)
    raise RuntimeError("Gemini failed after 5 attempts")


def mimo_call(prompt, model="mimo-v2.5-pro", temperature=0.0, system=None,
              base_url=None):
    """调用<proprietary-vendor> MiMo API（OpenAI 兼容，Bearer header，专属套餐 sgp 网关）

    NOTE 2026-05-13: 套餐 token 走 <proprietary-api-host> 而非公网
    <proprietary-api-host>；header 必须为 Authorization: Bearer，模型名必须全小写。
    旧实现 (api-key header + <proprietary-api-host>) 在新 token 上恒返 401。
    """
    api_key = os.environ.get("MIMO_API_KEY", "")
    if not api_key:
        raise RuntimeError("MIMO_API_KEY not set")
    base = (base_url
            or os.environ.get("MIMO_BASE_URL")
            or "https://<proprietary-api-host>/v1")
    url = base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
    }
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    payload = {
        "model": model.lower(),  # 强制小写：MiMo-V2.5-Pro 会被 400 拒绝
        "messages": msgs,
        "max_completion_tokens": 2048,
        "temperature": temperature,
        "top_p": 0.95,
        "stream": False,
    }
    for attempt in range(5):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            wait = min(60, 3 * (2 ** attempt))
            logger.warning("MiMo attempt %d failed: %s, waiting %ds", attempt + 1, str(e), wait)
            time.sleep(wait)
    raise RuntimeError("MiMo failed after 5 attempts")


def call_llm(prompt, provider="ollama", model=None, temperature=0.0):
    """统一 LLM 调用接口"""
    if provider == "ollama":
        m = model or "qwen2.5:3b"
        return ollama_call(prompt, model=m, temperature=temperature)
    elif provider == "groq":
        m = model or "llama-3.3-70b-versatile"
        return groq_call(prompt, model=m, temperature=temperature)
    elif provider == "gemini":
        return gemini_call(prompt, temperature=temperature)
    elif provider == "mimo":
        m = model or "mimo-v2.5-pro"
        return mimo_call(prompt, model=m, temperature=temperature)
    else:
        raise ValueError("Unknown provider: " + provider)


def call_with_fallback(prompt, providers=None, temperature=0.0):
    """尝试多个 provider，返回 (provider_name, response)"""
    if providers is None:
        providers = ["ollama", "groq", "gemini"]
    last_err = None
    for p in providers:
        try:
            resp = call_llm(prompt, provider=p, temperature=temperature)
            return p, resp
        except Exception as e:
            last_err = e
            logger.warning("Provider %s failed: %s", p, str(e))
            time.sleep(3)
    raise RuntimeError("All providers failed: " + str(last_err))


def batch_call(prompts, out_path, provider="ollama", min_interval=2.0):
    """批量调用 LLM，支持断点续传"""
    done = set()
    if os.path.exists(out_path):
        with open(out_path, 'r') as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done.add(rec["id"])
    
    with open(out_path, 'a') as f:
        for i, prompt in enumerate(prompts):
            if i in done:
                continue
            t0 = time.time()
            try:
                resp = call_llm(prompt, provider=provider)
                record = {"id": i, "prompt": prompt[:200], "response": resp}
            except Exception as e:
                record = {"id": i, "prompt": prompt[:200], "error": str(e)}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            dt = time.time() - t0
            if dt < min_interval:
                time.sleep(min_interval - dt)
            if (i + 1) % 50 == 0:
                logger.info("Progress: %d/%d", i + 1, len(prompts))
    
    logger.info("Batch complete: %s", out_path)
