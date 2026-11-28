# basic_llm

LoRA fine-tuning of [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) on the Dolly instruction dataset.

## Goal

A learning project: build the core pieces of LLM fine-tuning by hand.

- **LoRA from scratch**: a custom `LoRALinear` layer ([src/lora.py](src/lora.py)), with no `peft`
- **Supervised fine-tuning (SFT)**: a plain PyTorch training loop where the loss covers only the response tokens
- **Evaluation**: validation loss and perplexity, on the response only and on the full sequence
- **Behavior check**: a simple verifier ([src/verifyer.py](src/verifyer.py)) that tests whether the model picked up a target behavior (ending responses with `§`)

## Setup

```bash
uv sync
```

## Train

Settings live in `config.py`.

```bash
python train.py
```

## Test

```bash
pytest
```
