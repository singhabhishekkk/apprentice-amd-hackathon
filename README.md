<div align="center">

<img src="assets/logo.svg" width="96" alt="Apprentice logo">

# Apprentice

**The apprentice watches the expensive model work. Then takes over.**

[![website](https://img.shields.io/badge/runapprentice.com-visit-EDE6D6)](https://runapprentice.com)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![built on](https://img.shields.io/badge/built%20on-AMD%20MI300X%20%C2%B7%20ROCm-ED1C24)](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
[![powered by](https://img.shields.io/badge/powered%20by-Gemma%204%20E4B-4285F4)](https://ai.google.dev/gemma)
[![fallback](https://img.shields.io/badge/fallback-Fireworks%20AI-8B5CF6)](https://fireworks.ai)
[![every number](https://img.shields.io/badge/every%20number-measured-orange)](https://github.com/singhabhishekkk/apprentice-benchmark)

**AMD Developer Hackathon: ACT II · Track 3 (Unicorn Track)**

*Demo film: link added at submission (the 4K cut with the AMD + Gemma beat).*

</div>

[Apprentice](https://runapprentice.com) turns a repeatable frontier-LLM task into a small
model you own: capture real input/output pairs, verify them to gold, optimize the prompt,
fine-tune a small open model, and let it take traffic **only after it passes your own
held-out eval**. Rollback is one env var.

This repo is the hackathon slice: the same loop, with **Gemma 4 E4B trained and served on
AMD**, and a router that answers locally first and falls back to Fireworks AI, with every
fallback counted against the savings.

## What runs where

| Piece | Where | What it does |
|---|---|---|
| `scripts/train_gemma_rocm.py` | AMD MI300X pod (ROCm) | bf16 LoRA fine-tune of `google/gemma-4-E4B-it` (TRL + PEFT), scores a held-out slice before and after |
| vLLM | same AMD pod | serves the fine-tuned Gemma behind an OpenAI-compatible endpoint |
| `router/main.py` | anywhere (containerized) | local-first routing, deterministic JSON gate, Fireworks fallback, honest tally at `/report` |
| `data/prepare_data.py` | anywhere | rebuilds the exact golden dataset (CUAD v1, 200 rows, seed 42) used by the public benchmark |

## Results

Measured on the AMD pod, 2026-07-10 (`run_report.json`, printed by the run -- this repo
never carries a number that was not measured). Held-out: the same 60 CUAD rows and
field-level F1 as the public benchmark.

| Measurement | Value |
|---|---|
| Gemma 4 E4B raw (no fine-tune), held-out F1 | 36.33 |
| Gemma 4 E4B fine-tuned (bf16 LoRA, 3 epochs), held-out F1 | **61.67** |
| Train wall time on the AMD GPU | 505.8 s (~8.4 GPU-minutes per adapter) |
| Baseline / final eval wall time | 556.2 s / 1079.8 s |

The run itself is in the repo: [`notebooks/mi300x-run.ipynb`](notebooks/mi300x-run.ipynb)
is the actual pod notebook with outputs unedited, and
[`notebooks/run_report.json`](notebooks/run_report.json) is the machine-readable report
it printed.

For scale: the best teacher result published for this task is gpt-5.4-mini at 34.00
baseline / 36.33 GEPA-optimized ([benchmark README](https://github.com/singhabhishekkk/apprentice-benchmark/tree/main/tasks/contract-clause-extraction)).
The AMD-trained Gemma E4B fine-tune scores +25.3 over the optimized teacher on the same
held-out rows. Honest notes: this run is bf16 TRL + PEFT (not 4-bit), and
`torch.cuda.get_device_name` returns an empty string under this ROCm build, so the
report says "AMD GPU" while `rocm-smi` (device 0x744b, GPU% pinned at 100 during
training) is the hardware proof.

## Reproduce

**1. Data (anywhere):**

```bash
python data/prepare_data.py   # downloads CUAD v1, writes golden.csv (200 rows, seed 42)
```

**2. Train on the AMD pod:**

```bash
docker build -f Dockerfile.train -t apprentice-train .
docker run --device=/dev/kfd --device=/dev/dri --group-add video \
  -v $PWD/out:/work/out apprentice-train
# prints raw + fine-tuned held-out scores, saves adapter + run_report.json
```

**3. Serve the fine-tune with vLLM on the same pod:**

```bash
vllm serve google/gemma-4-E4B-it --enable-lora \
  --lora-modules gemma-cuad-lora=out/adapter --port 8000
```

**4. Run the router (containerized):**

```bash
docker build -t apprentice-router .
docker run -p 8900:8900 \
  -e LOCAL_BASE_URL=http://<pod-ip>:8000/v1 \
  -e LOCAL_MODEL=gemma-cuad-lora \
  -e FIREWORKS_API_KEY=... \
  -e FIREWORKS_MODEL=accounts/fireworks/models/<allowlisted-model> \
  -e REQUIRED_KEYS=document_name,parties,agreement_date,governing_law,anti_assignment \
  apprentice-router
```

```bash
curl -s localhost:8900/route -X POST -H 'Content-Type: application/json' \
  -d '{"prompt": "<contract excerpt prompt>"}'
curl -s localhost:8900/report   # local share, fallbacks, tokens: the honest tally
```

**Rollback demo:** restart with `-e LOCAL_ENABLED=false` and every request goes to
Fireworks. That is the whole rollback story: one env var.

**No AMD pod handy?** The router runs everywhere on its own: set
`LOCAL_ENABLED=false` and it serves 100% from Fireworks while `/report` shows the
rollback state honestly. Only a Fireworks API key is needed:

```bash
docker run -p 8900:8900 -e LOCAL_ENABLED=false \
  -e FIREWORKS_API_KEY=... \
  -e FIREWORKS_MODEL=accounts/fireworks/models/<allowlisted-model> \
  apprentice-router
curl -s localhost:8900/health   # {"status":"ok","local_enabled":false}
```

## Why the gate matters (the product thesis)

A cheap model that silently degrades quality is a loss dressed as a saving. The router's
gate is deterministic (valid JSON, required keys present); anything else falls back to
Fireworks, and the fallback is **counted against the savings** at `/report`. The full
product applies the same rule with human-verified gold rows and a canary rollout:
[runapprentice.com](https://runapprentice.com).

## The wider Apprentice surface

| Piece | Where | What |
|---|---|---|
| Product | [runapprentice.com](https://runapprentice.com) | the full loop: capture, verify, optimize, train, eval gate, canary, rollback |
| Docs | [docs.runapprentice.com](https://docs.runapprentice.com) | quickstart, deploy guides (Mac/MLX, Kubernetes/vLLM), SDK reference |
| Benchmark | [apprentice-benchmark](https://github.com/singhabhishekkk/apprentice-benchmark) | 3 public tasks, every number rerunnable (Apache-2.0) |
| Agent skill | [apprentice-skill](https://github.com/singhabhishekkk/apprentice-skill) | Claude Code plugin, Codex plugin, Copilot CLI skill (MIT) |
| SDK | [runapprentice on PyPI](https://pypi.org/project/runapprentice/) | `pip install runapprentice` · free local optimize and MLX train, no account needed |

## License

MIT. Gemma 4 is Apache-2.0; this project carries a "Built with Gemma" acknowledgment.
CUAD v1 is CC-BY-4.0 (TheAtticusProject).
