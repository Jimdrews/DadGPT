# DadGPT

[![CI](https://github.com/Jimdrews/DadGPT/actions/workflows/ci.yml/badge.svg)](https://github.com/Jimdrews/DadGPT/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus)
![CUDA](https://img.shields.io/badge/CUDA-12%20%7C%2013-76B900?logo=nvidia)

A GPT-style transformer written from scratch in C++20 and CUDA that writes dad jokes. It can
train its own small model or fine-tune OpenAI's GPT-2 (124M).

There's no PyTorch and no Python at runtime. The tokenizers, forward and backward passes,
optimizer, attention kernels and sampling are all written by hand.

## Example output

These come from the default `small` model (~25M parameters) after 800 training steps on the
dad-jokes dataset. They're unedited:

```text
> Why don't skeletons fight each other?
Because they are too cools

> Why don't skeletons fight each other?
Because they can get a broom

> I told my wife
I had a gravel

> What do you call a fish with no eyes?
A page

> Why did the scarecrow win an award?
Because he was drink
```

It's getting there. For actual punchlines, [fine-tune GPT-2](#fine-tune-gpt-2).

## What's inside

- **Hand-written CUDA kernels**: fused causal (flash-style) attention, softmax, layer norm,
  GELU and dropout, with cuBLAS bf16 tensor-core matmuls.
- **A complete training loop**: AdamW, warmup plus cosine LR, gradient accumulation, dropout,
  periodic eval, and checkpoints that resume automatically.
- **Fast generation**: a KV cache, batched prompt prefill, temperature, top-k and repetition
  penalty.
- **Two tokenizers**: a custom byte-level BPE trained on the jokes, and OpenAI's GPT-2 BPE.
- **Loads Hugging Face GPT-2** directly from `model.safetensors`, checked against
  `transformers` in the tests.
- **Gradient-check tests** run finite differences over the whole model, with and without fused
  attention and dropout. There are also compute-sanitizer memcheck and racecheck targets.

## Requirements

- An NVIDIA GPU. The build targets sm_86 (RTX 30xx) by default; change it with
  `-DCMAKE_CUDA_ARCHITECTURES=<sm>`.
- CUDA Toolkit, CMake and Ninja
- [vcpkg](https://github.com/microsoft/vcpkg), with `VCPKG_ROOT` set
- A C++20 compiler: MSVC 2022 on Windows or GCC 14 on Linux

## Build

```sh
cmake --preset windows-release        # Linux: release
cmake --build --preset windows-release
```

On Windows, run these from an "x64 Native Tools" developer prompt. The executable ends up at
`build/<preset>/src/dad-gpt`. Use the `windows` / `default` presets for debug builds.

## Quick start: train from scratch

1. Download `train.csv` and `test.csv` from the
   [shuttie/dadjokes](https://huggingface.co/datasets/shuttie/dadjokes) dataset into `data/`.
2. Tokenize, train and sample:

```sh
dad-gpt prepare
dad-gpt train
dad-gpt generate --prompt "Why did the scarecrow win an award?"
```

Training writes `data/checkpoint_step_N.bin` and picks up from the latest one when it's rerun.

## Fine-tune GPT-2

1. Put the [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) files
   (`vocab.json`, `merges.txt`, `config.json`, `model.safetensors`) in `data/gpt2`.
2. Copy `train.csv` and `test.csv` into a new directory, e.g. `data/jokes_gpt2`, and run:

```sh
dad-gpt prepare data/jokes_gpt2 --tokenizer gpt2
dad-gpt train --data data/jokes_gpt2 --init-from data/gpt2
dad-gpt generate --data data/jokes_gpt2 --prompt "What do you call a fish with no eyes?"
```

`dad-gpt generate --hf data/gpt2` samples from stock GPT-2 without any training.

## Commands

| Command | Common options |
|---|---|
| `prepare [dir]` | `--tokenizer dadbpe\|gpt2` |
| `train` | `--data DIR`, `--model NAME`, `--init-from DIR`, `--steps N`, `--batch N`, `--accum N`, `--lr F` |
| `generate` | `--data DIR`, `--checkpoint PATH`, `--prompt TEXT`, `--count N`, `--temp F` (0 = greedy), `--top-k N` |

Run `dad-gpt` with no arguments for the full list of options.

## Model presets

| `--model` | Size | Use |
|---|---|---|
| `tiny` | ~0.5M params | smoke tests |
| `small` (default) | ~25M params | the dad-jokes model |
| `gpt2-124m` | 124M params | GPT-2 small; use with `--init-from` |

## Tests

```sh
ctest --preset windows                # Linux: default
```

The tests need a CUDA GPU. The GPT-2 tests also need the files in `data/gpt2` and are skipped
without them. To regenerate their reference fixture with Hugging Face `transformers`:

```sh
uv run --script scripts/gpt2_reference.py data/gpt2 tests/fixtures/gpt2_reference.json
```

## Project layout

| Path | Contents |
|---|---|
| `src/` | tokenizers, data loading, model, backward pass, optimizer, training, generation, CLI |
| `cuda/` | GPU kernels and cuBLAS wrappers, including fused attention |
| `tests/` | GoogleTest suite, gradient checks and op benchmarks |
| `scripts/` | Python script that builds the GPT-2 reference fixture |

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

- [GPT-2](https://huggingface.co/openai-community/gpt2) by OpenAI
- The [shuttie/dadjokes](https://huggingface.co/datasets/shuttie/dadjokes) dataset, which
  comes from Reddit's r/dadjokes
- Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and
  [llm.c](https://github.com/karpathy/llm.c), for showing that this is a reasonable thing to do

## License

[Apache-2.0](LICENSE). See [NOTICE](NOTICE) for third-party attributions.
