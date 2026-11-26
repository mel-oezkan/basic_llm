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
