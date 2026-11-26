import logging

logger = logging.getLogger(__name__)


def count_params(model, prefix="") -> None:
    """Log the number of trainable and total parameters in the model."""

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    label = f"{prefix}: " if prefix else ""
    logger.info(
        f"{label}trainable: {trainable:,} / total: {total:,} ({100 * trainable / total:.4f}%)"
    )
