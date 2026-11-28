import torch

from src.helper import token_logprobs


def masked_nll(
    log_probs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the negative log-likelihood loss with a mask.

    params:
        log_probs: Tensor of shape (B, T-1)
        labels: Tensor of shape (B, T)
        mask: Tensor of shape (B, T)
    return:
        loss: Tensor of shape (1,)
        sum: Tensor of shape (1,)
    """
    m = mask[:, 1:].float()
    return -(log_probs * m).sum(), m.sum()


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
    nll, m_sum = masked_nll(lp, labels, attention_mask)
    return nll / m_sum.clamp(min=1)
