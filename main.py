import polars as pl
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import  AutoModelForCausalLM

from src.lora import add_lora, count_params
from src.helper import get_dataloaders, token_logprobs
from src.tokenizer_utils import load_tokenizer, load_splits

import logging
import config as cfg
from src.training.utils import validate_training
from src.uitls.logging import setup_logging
from src.models.utils import count_params

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
    # prepate script
    setup_logging(logfile="logs/train.log")

    # load the data
    data_dirs = {
        "train": "data/dolly/train.jsonl",
        "val": "data/dolly/val.jsonl",
        "test": "data/dolly/test.jsonl",
    }
    tok = load_tokenizer(cfg.MODEL)
    ds = load_splits(tok, data_dirs)
    dataloaders = get_dataloaders(tok, ds)

    # load the model and add LoRA
    model = AutoModelForCausalLM.from_pretrained(cfg.MODEL, device_map="auto")
    model = add_lora(model, r=4, alpha=1.0)
    count_params(model, prefix="LoRA params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR)
    validate_training(model, dataloaders["val"], sft_loss)

    # train the model
    for epoch in range(10):
        model.train()

        for batch in dataloaders["train"]:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            loss_mask = batch["loss_mask"].to(model.device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = sft_loss(logits, input_ids, loss_mask)
            loss.backward()
            optimizer.step()
            logging.info(f"Epoch: {epoch}, Batch: {batch['input_ids'].shape[0]}, Loss: {loss.item()}")
        
        model.eval()
        for batch in dataloaders["val"]:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            loss_mask = batch["loss_mask"].to(model.device)

            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        
            val_loss = sft_loss(logits, input_ids, loss_mask)
            logging.info(f"Epoch: {epoch}, Batch: {batch['input_ids'].shape[0]}, Loss: {val_loss.item()}")

    



