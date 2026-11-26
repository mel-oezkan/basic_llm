import torch
import torch.nn as nn
from datasets import DatasetDict
from torch.utils.data import DataLoader

from src.tokenizer_utils import collate, load_tokenizer


def token_logprobs(
    logits: torch.Tensor, 
    input_ids: torch.Tensor
) -> torch.Tensor:
    """Helper function to compute the logprobs given the logits and input_ids."""
    log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    targets = input_ids[:, 1:]
    return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def get_logprobs(
    model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """ Compute the logprobs of the response tokens given the attn_mask.

    params:
        model: The transformer model
        input_ids: Tensor of shape (B, T)
        attention_mask: Tensor of shape (B, T)
    return: 
        Tensor of the logprobs of the response (B, T-1)
    """
    # prevents left-padding from shifting the inputs
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    logits = model(
        input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids
    ).logits
    return token_logprobs(logits, input_ids)


def sequence_logprob(
    model, prompt_ids: list[list[int]], response_ids: list[list[int]]
) -> float:
    """Compute the logprob over the response ids.

    params:
        model: The transformer model
        prompt_ids: List of variable length ints
        response_ids: List of variable lenght ints

    return: Sum of the response logprobs.
    """
    # combine the promt and reponse to find longest sequence (derive padd count)
    n_inputs = len(prompt_ids)
    if n_inputs != len(response_ids):
        raise ValueError("number of prompts and responses does not match")

    max_tokens = max(len(a) + len(b) for a, b in zip(prompt_ids, response_ids))

    # create the input and pad the content + masks
    PAD_ID = 0
    batch_input = torch.full((n_inputs, max_tokens), PAD_ID, dtype=torch.long)
    batch_mask = torch.zeros((n_inputs, max_tokens), dtype=torch.long)
    #! if this script is changed to a different model we would need to check that 0 is a pad token
    response_mask = torch.zeros((n_inputs, max_tokens - 1), dtype=torch.bool)

    for i in range(n_inputs):
        assert len(response_ids[i]) > 0, "There is no response"
        total_len = len(prompt_ids[i]) + len(response_ids[i])

        fill_value = prompt_ids[i] + response_ids[i]
        batch_input[i, -total_len:] = torch.tensor(fill_value)
        batch_mask[i, -total_len:] = 1
        response_mask[i, -len(response_ids[i]) :] = True

    log_probs = get_logprobs(model, batch_input, batch_mask)
    return (log_probs * response_mask).sum(-1)



def get_dataloaders(tokenizer, dd: DatasetDict) -> dict[str, DataLoader]:
    """Create dataloaders for the train, val, and test splits.

    params:
        dd: DatasetDict with "train", "val", and "test" splits

    return:
        dict of dataloaders
    """
    return {
        split: DataLoader(
            ds,
            batch_size=8,
            shuffle=split == "train",
            collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        )
        for split, ds in dd.items()
    }
