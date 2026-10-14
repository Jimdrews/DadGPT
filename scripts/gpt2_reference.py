# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.40", "safetensors"]
# ///
"""Writes the reference fixture for DadGPT's GPT-2 tests using Hugging Face transformers.

    uv run --script scripts/gpt2_reference.py data/gpt2 tests/fixtures/gpt2_reference.json

data/gpt2 must hold the openai-community/gpt2 files: vocab.json, merges.txt, config.json and
model.safetensors. The fixture records HF's tokenization of edge-case strings, per-position
logit summaries for a prompt, and a greedy continuation, so the C++ tests can check the
tokenizer, the weight loader and the forward pass against an independent implementation.
"""

import json
import sys
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

TOKENIZE_CASES = [
    "Hello world",
    "Hello, I'm a language model,",
    "  leading spaces and trailing   ",
    "It's 2024!!! café naïve 日本語 \U0001f916 emoji",
    "\n\nnewlines\tand\ttabs \r\n windows",
    "don't won't I'll you've they're he'd we'm 's",
    "3.14159 1,000,000 x2 V8 A100",
    "Why did the scarecrow win an award?\nBecause he was outstanding in his field.",
]
PROMPT = "Hello, I'm a language model,"
PROBE_IDS = [0, 11, 13, 198, 262, 290, 318, 340, 50256]
GREEDY_TOKENS = 12


def main() -> None:
    model_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    tok = GPT2Tokenizer(str(model_dir / "vocab.json"), str(model_dir / "merges.txt"))
    model = GPT2LMHeadModel.from_pretrained(str(model_dir), torch_dtype=torch.float32).eval()

    fixture = {"tokenize": [{"text": t, "ids": tok.encode(t)} for t in TOKENIZE_CASES]}

    ids = tok.encode(PROMPT)
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0].double()  # [T, vocab]
    positions = []
    for t in range(len(ids)):
        row = logits[t]
        top = torch.topk(row, 5)
        positions.append(
            {
                "argmax": int(row.argmax()),
                "logsumexp": float(torch.logsumexp(row, 0)),
                "probes": [float(row[i]) for i in PROBE_IDS],
                "top5_ids": [int(i) for i in top.indices],
                "top5_logits": [float(v) for v in top.values],
            }
        )
    fixture["logits"] = {"prompt": PROMPT, "ids": ids, "probe_ids": PROBE_IDS, "positions": positions}

    with torch.no_grad():
        out = model.generate(
            torch.tensor([ids]), max_new_tokens=GREEDY_TOKENS, do_sample=False, pad_token_id=50256
        )
    fixture["greedy"] = {"prompt_ids": ids, "continuation": out[0, len(ids):].tolist()}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixture, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {out_path}: greedy -> {tok.decode(fixture['greedy']['continuation'])!r}")


if __name__ == "__main__":
    main()
