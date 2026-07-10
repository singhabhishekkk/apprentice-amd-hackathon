"""Fine-tune Gemma 4 E4B with a bf16 LoRA on AMD (ROCm), TRL + PEFT.

The AMD-hosted half of the Apprentice hackathon entry: the same golden
dataset and field-level-F1 metric as the public benchmark
(github.com/singhabhishekkk/apprentice-benchmark), trained on an MI300X
instead of Colab. bf16, no 4-bit: 192 GB HBM makes quantization pointless
for a 4B LoRA, and bf16 is the well-trodden ROCm path.

Run (inside the ROCm PyTorch container, see Dockerfile.train):
    python scripts/train_gemma_rocm.py --data data/golden.csv --output-dir out/adapter

Every score this prints is measured; nothing here fabricates a number.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = "google/gemma-4-E4B-it"
SEED = 42
SPLIT = 0.7  # same 140/60 split as every published Apprentice run


def json_field_f1(expected_str: str, actual_str: str) -> float:
    """Tier-1 metric, byte-for-byte the benchmark's json_field_metric."""
    try:
        expected = json.loads(expected_str)
        actual = json.loads(actual_str)
    except json.JSONDecodeError:
        return 0.0
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return 0.0
    tp = sum(1 for k, v in expected.items() if actual.get(k) == v)
    precision = tp / len(actual) if actual else 0.0
    recall = tp / len(expected) if expected else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def extract_json(s: str) -> str:
    s = re.sub(r"<think>.*?(</think>|$)", "", s, flags=re.DOTALL)
    s = re.sub(r"```(json)?", "", s)
    m = re.search(r"\{.*\}", s, re.DOTALL)
    return m.group(0) if m else s


def load_rows(path: Path) -> tuple[list[dict], list[dict]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [{"text": r["input"], "expected": r["output"]} for r in csv.DictReader(f)]
    random.Random(SEED).shuffle(rows)
    split = int(len(rows) * SPLIT)
    return rows[:split], rows[split:]


def evaluate(model, tokenizer, devset: list[dict]) -> float:
    model.eval()
    scores = []
    for row in devset:
        messages = [{"role": "user", "content": row["text"]}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # text= keyword, not positional: Gemma 4 ships a multimodal Processor
        # and a positional arg is interpreted as an image.
        inputs = tokenizer(text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        scores.append(json_field_f1(row["expected"], extract_json(completion)))
    return 100 * sum(scores) / len(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="golden.csv with input,output columns")
    parser.add_argument("--output-dir", type=Path, default=Path("out/adapter"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--skip-baseline", action="store_true", help="skip the pre-finetune eval")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "No GPU visible. Under ROCm this must be true on an MI300X pod."
    # Some ROCm builds return an empty device name; rocm-smi is the real proof.
    device_name = torch.cuda.get_device_name(0) or "AMD GPU (name unavailable in this ROCm build)"
    print(f"device: {device_name}")

    trainset, devset = load_rows(args.data)
    print(f"train {len(trainset)}, held-out {len(devset)} (seed {SEED})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")

    report: dict = {"device": device_name, "model": MODEL_ID, "seed": SEED}

    if not args.skip_baseline:
        t0 = time.time()
        report["raw_score"] = round(evaluate(model, tokenizer, devset), 2)
        report["baseline_eval_seconds"] = round(time.time() - t0, 1)
        print(f"raw (no fine-tune): {report['raw_score']:.2f}")

    lora = LoraConfig(
        r=16,
        lora_alpha=16,
        # Regex, not a name list: Gemma 4 is multimodal and its vision/audio
        # towers wrap linears in Gemma4ClippableLinear, which PEFT cannot
        # adapt. Scoping to the language model adapts only real nn.Linear
        # projections -- this is a text-only fine-tune anyway.
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    def to_text(row: dict) -> dict:
        messages = [
            {"role": "user", "content": row["text"]},
            {"role": "assistant", "content": row["expected"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    train_ds = Dataset.from_list([to_text(r) for r in trainset])
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            logging_steps=5,
            bf16=True,
            output_dir=str(args.output_dir / "checkpoints"),
            report_to="none",
            # TRL refuses packing for vision-language models, and Gemma 4
            # counts as one even in this text-only run. Padding waste is
            # acceptable at 140 rows.
            packing=False,
        ),
    )
    t0 = time.time()
    trainer.train()
    report["train_seconds"] = round(time.time() - t0, 1)

    t0 = time.time()
    report["finetuned_score"] = round(evaluate(model, tokenizer, devset), 2)
    report["final_eval_seconds"] = round(time.time() - t0, 1)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    report_path = args.output_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("=" * 52)
    if "raw_score" in report:
        print(f"Gemma 4 E4B raw        : {report['raw_score']:.2f}")
    print(f"Gemma 4 E4B fine-tuned : {report['finetuned_score']:.2f}")
    print(f"train wall time        : {report['train_seconds']}s on {device_name}")
    print("=" * 52)
    print(f"adapter + report -> {args.output_dir}")


if __name__ == "__main__":
    main()
