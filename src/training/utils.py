import logging

import torch

logger = logging.getLogger(__name__)


def validate_training(model, dataloader, loss_fn) -> None:
    """Single batch validation run to check if the model is training correctly."""

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            loss_mask = batch["loss_mask"].to(model.device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = loss_fn(logits, input_ids, loss_mask)
            logger.info(f"Sanity check passed, loss: {loss.item():.4f}")
            break


def config_to_dict(cfg) -> dict:
    """Collect the UPPERCASE settings from a config module into a plain dict."""
    return {
        k: v
        for k, v in vars(cfg).items()
        if k.isupper() and isinstance(v, (str, int, float, bool, type(None)))
    }


def save_lora(model, path: str, cfg) -> None:
    """Save the trainable (LoRA) weights together with the config used to train them."""
    torch.save(
        {
            "config": config_to_dict(cfg),
            "state_dict": {
                n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad
            },
        },
        path,
    )
    logger.info(f"Saved LoRA weights and config to {path}")
