import polars as pl
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader

from src.helper import token_logprobs

ds = load_dataset(
    "json",
    data_files={
        "train": "data/dolly/train.jsonl",
        "val": "data/dolly/val.jsonl",
        "test": "data/dolly/test.jsonl",
    },
)


def sft_loss(
    logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Compute the SFT loss given the logits, labels, and attention mask.

    params:
        logits: Tensor of shape (B, T, V)
        labels: Tensor of shape (B, T)
        attention_mask: Tensor of shape (B, T)
    """
    lp = token_logprobs(logits, labels)
    m = attention_mask[:, 1:].float()
    return -(lp * m).sum() / m.sum().clamp(min=1)


if __name__ == "__main__":
    # load the dataset

    pass
