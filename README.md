# Apprentice — AMD-hosted Gemma, eval-gated takeover

**AMD Developer Hackathon: ACT II · Track 3 (Unicorn Track)**

Built on **AMD MI300X (ROCm)** · Powered by **Gemma 4 E4B** · **Fireworks AI** as the honest fallback

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

Filled from the real MI300X run's `run_report.json` before submission. This repo never
carries a number that was not measured. The published CUDA/Colab numbers for the same
loop live in [apprentice-benchmark](https://github.com/singhabhishekkk/apprentice-benchmark).

| Measurement | Value |
|---|---|
| Gemma 4 E4B raw (no fine-tune), held-out F1 | *pending MI300X run* |
| Gemma 4 E4B fine-tuned, held-out F1 | *pending MI300X run* |
| Train wall time on MI300X | *pending* |
| GPU-minutes per adapter | *pending* |

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

## Why the gate matters (the product thesis)

A cheap model that silently degrades quality is a loss dressed as a saving. The router's
gate is deterministic (valid JSON, required keys present); anything else falls back to
Fireworks, and the fallback is **counted against the savings** at `/report`. The full
product applies the same rule with human-verified gold rows and a canary rollout:
[runapprentice.com](https://runapprentice.com).

## The wider Apprentice surface

- Product: [runapprentice.com](https://runapprentice.com) · [docs](https://docs.runapprentice.com)
- Reproducible benchmark (3 tasks, every number rerunnable): [apprentice-benchmark](https://github.com/singhabhishekkk/apprentice-benchmark)
- Agent skill (Claude Code / Codex / Copilot CLI): [apprentice-skill](https://github.com/singhabhishekkk/apprentice-skill)
- SDK: `pip install runapprentice`

## License

MIT. Gemma 4 is Apache-2.0; this project carries a "Built with Gemma" acknowledgment.
CUAD v1 is CC-BY-4.0 (TheAtticusProject).
