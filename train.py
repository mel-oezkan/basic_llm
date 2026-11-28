import logging
import math

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

import config as cfg
from src.helper import get_dataloaders, token_logprobs
from src.lora import apply_lora
from src.models.utils import count_params
from src.tokenizer_utils import load_splits, load_tokenizer
from src.training.loss import masked_nll, sft_loss
from src.training.utils import save_lora, validate_training
from src.uitls.logging import setup_logging

logger = logging.getLogger(__name__)


def main():
    # prepate script
    setup_logging(logfile="logs/train.log")
    try:
        model_dtype = getattr(torch, cfg.MODEL_DTYPE)
        logging.info(f"Using model dtype: {model_dtype}")
    except AttributeError:
        model_dtype = None
        logging.warning(f"Invalid model dtype: {cfg.MODEL_DTYPE}. Using default dtype.")

    # load the data
    data_dirs = {
        "train": "data/dolly/train.jsonl",
        "val": "data/dolly/val.jsonl",
        "test": "data/dolly/test.jsonl",
    }
    tok = load_tokenizer(cfg.MODEL, cfg.TEMPLATE_FILE)
    ds = load_splits(tok, data_dirs)
    dataloaders = get_dataloaders(tok, ds)

    # load the model and add LoRA
    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL, device_map="cuda", dtype=model_dtype
    )
    model = apply_lora(
        model, r=cfg.LORA_R, alpha=cfg.LORA_ALPHA, lora_dtype=cfg.LORA_DTYPE
    )
    count_params(model, prefix="LoRA params")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.LR
    )
    validate_training(model, dataloaders["val"], sft_loss)

    # train the model
    for epoch in tqdm(range(cfg.EPOCHS), desc="Training"):
        model.train()

        n_batches = len(dataloaders["train"])
        train_loss = 0.0
        for step, batch in tqdm(
            enumerate(dataloaders["train"]), desc="Training", total=n_batches
        ):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            loss_mask = batch["loss_mask"].to(model.device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = sft_loss(logits, input_ids, loss_mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        logger.info(
            f"epoch {epoch} | batch {step + 1}/{n_batches} | train loss {train_loss / n_batches:.4f}"
        )

        model.eval()
        val_nll, val_tokens = 0.0, 0
        val_full_nll, val_full_tokens = 0.0, 0

        for batch in tqdm(dataloaders["val"], desc="Validation"):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            loss_mask = batch["loss_mask"].to(model.device)

            with torch.no_grad():
                logits = model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits

                log_probs = token_logprobs(logits, input_ids)

                resp_nll, resp_n = masked_nll(log_probs, input_ids, loss_mask)
                full_nll, n = masked_nll(log_probs, input_ids, attention_mask)

            val_nll += resp_nll.item()
            val_full_nll += full_nll.item()

            val_tokens += resp_n
            val_full_tokens += n

        val_loss = val_nll / val_tokens.clamp(min=1)
        val_full_loss = val_full_nll / val_full_tokens.clamp(min=1)

        val_ppl = math.exp(val_loss)
        val_full_ppl = math.exp(val_full_loss)

        logger.info(
            f"epoch {epoch} | val loss {val_loss:.4f} | val ppl {val_ppl:.2f} | val full ppl {val_full_ppl:.2f}"
        )

        if (epoch + 1) % cfg.SAVE_EVERY == 0:
            save_lora(model, f"/kaggle/working/lora_epoch_{epoch + 1}.pt", cfg)

    save_lora(model, "/kaggle/working/lora.pt", cfg)


if __name__ == "__main__":
    main()
