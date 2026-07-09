"""Apprentice router: local Gemma first, Fireworks fallback, honest accounting.

The takeover pattern from runapprentice.com in one file. Requests go to the
locally served fine-tuned Gemma (vLLM on the same AMD pod). If the local
answer fails the deterministic gate (invalid JSON or missing required keys),
the request falls back to Fireworks AI, and that fallback is COUNTED AGAINST
the savings. Rollback is one env var: LOCAL_ENABLED=false sends 100% of
traffic to Fireworks.

Run:
    uvicorn router.main:app --host 0.0.0.0 --port 8900

Env:
    LOCAL_BASE_URL      default http://127.0.0.1:8000/v1  (vLLM on the pod)
    LOCAL_MODEL         served model or LoRA name in vLLM
    FIREWORKS_API_KEY   required for fallback
    FIREWORKS_MODEL     e.g. accounts/fireworks/models/<allowlisted-model>
    LOCAL_ENABLED       "false" = rollback, everything goes to Fireworks
    REQUIRED_KEYS       comma-separated JSON keys the answer must contain
"""

from __future__ import annotations

import json
import os
import re
import time

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

LOCAL_BASE_URL = os.environ.get("LOCAL_BASE_URL", "http://127.0.0.1:8000/v1")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "gemma-cuad-lora")
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_MODEL = os.environ.get("FIREWORKS_MODEL", "")
REQUIRED_KEYS = [k for k in os.environ.get("REQUIRED_KEYS", "").split(",") if k]


def local_enabled() -> bool:
    return os.environ.get("LOCAL_ENABLED", "true").lower() != "false"


app = FastAPI(title="apprentice-router")

# In-memory tally. ponytail: one process, one demo; a DB would be theater here.
TALLY = {"local_served": 0, "fallbacks": 0, "remote_only": 0, "fireworks_tokens": 0}


class RouteRequest(BaseModel):
    prompt: str


def gate(completion: str) -> tuple[bool, str]:
    """Deterministic JSON gate: parseable object + every required key present."""
    m = re.search(r"\{.*\}", completion, re.DOTALL)
    if not m:
        return False, "no JSON object in output"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return False, f"invalid JSON: {e}"
    if not isinstance(obj, dict):
        return False, "JSON is not an object"
    missing = [k for k in REQUIRED_KEYS if k not in obj]
    if missing:
        return False, f"missing keys: {missing}"
    return True, "ok"


async def chat(base_url: str, model: str, prompt: str, api_key: str | None = None) -> tuple[str, int]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    tokens = data.get("usage", {}).get("total_tokens", 0)
    return data["choices"][0]["message"]["content"], tokens


@app.post("/route")
async def route(req: RouteRequest) -> dict:
    t0 = time.time()
    decisions: list[dict] = []

    if local_enabled():
        try:
            local_out, _ = await chat(LOCAL_BASE_URL, LOCAL_MODEL, req.prompt)
            ok, reason = gate(local_out)
            decisions.append({"target": "local", "gate": reason})
            if ok:
                TALLY["local_served"] += 1
                return {
                    "answer": local_out,
                    "served_by": "local",
                    "decisions": decisions,
                    "latency_seconds": round(time.time() - t0, 2),
                }
            TALLY["fallbacks"] += 1
        except httpx.HTTPError as e:
            decisions.append({"target": "local", "gate": f"transport error: {e}"})
            TALLY["fallbacks"] += 1
    else:
        decisions.append({"target": "local", "gate": "disabled (LOCAL_ENABLED=false, rollback active)"})
        TALLY["remote_only"] += 1

    remote_out, tokens = await chat(
        FIREWORKS_BASE_URL, FIREWORKS_MODEL, req.prompt, api_key=os.environ["FIREWORKS_API_KEY"]
    )
    TALLY["fireworks_tokens"] += tokens
    decisions.append({"target": "fireworks", "gate": "fallback served"})
    return {
        "answer": remote_out,
        "served_by": "fireworks",
        "decisions": decisions,
        "latency_seconds": round(time.time() - t0, 2),
    }


@app.get("/report")
def report() -> dict:
    total = TALLY["local_served"] + TALLY["fallbacks"] + TALLY["remote_only"]
    return {
        **TALLY,
        "total_requests": total,
        "local_share": round(TALLY["local_served"] / total, 3) if total else 0.0,
        # The rule that keeps the number honest: a fallback is a failed
        # replacement. Savings only accrue on requests the local model served.
        "note": "fallbacks and rollback traffic count against savings, never hidden",
        "rollback": not local_enabled(),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "local_enabled": local_enabled()}
