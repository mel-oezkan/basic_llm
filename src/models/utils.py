import logging

logging.basicConfig(level=logging.INFO)



def count_params(model, prefix="") -> None:
    """Print the number of trainable and total parameters in the model."""

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(
        f"{prefix}trainable: {trainable:,} / total: {total:,} ({100 * trainable / total:.4f}%)"
    )
