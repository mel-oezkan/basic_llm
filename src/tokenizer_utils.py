import torch
from transformers import AutoTokenizer


from datasets import DatasetDict
from pathlib import Path
from datasets import load_dataset

from config import SYSTEM



def load_tokenizer(name: str, template_file: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(name)
    tok.chat_template = Path(template_file).read_text()

    if tok.pad_token is None:
        tok.pad_token = "<|endoftext|>"
    return tok


def to_messages(ex: dict) -> list[dict]:
    """Takes the format of dolly row and turns it into the expectected format.

    Expected:
    [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": ex["prompt"]}
    ]
    """
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": ex["prompt"]},
    ]


def encode_prompt(messages: list[dict], tok: AutoTokenizer) -> list[int]:
    """Helper function to apply chat template.
    returns:
        Input ids of the chat template version of the messages
    """
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


def encode_completion(
    prompt_ids: list[int], completion: str, tok: AutoTokenizer, max_len: int
) -> dict:
    """Helper function to generate the full input ids of completion tasks."""
    comp_ids = tok(completion + "<|im_end|>", add_special_tokens=False)["input_ids"]

    ids = (prompt_ids + comp_ids)[:max_len]
    mask = ([0] * len(prompt_ids) + [1] * len(comp_ids))[:max_len]
    return {"input_ids": ids, "loss_mask": mask}


def tokenize(ex: dict, tok, max_len: int = 512) -> dict[str, list[int]]:
    """Dolly row -> {"input_ids": [...], "loss_mask": [...]}, unpadded, <= max_len.

    input_ids = templated prompt + response + "<|im_end|>".
    loss_mask is 1 on the response and "<|im_end|>", 0 on the prompt.
    """
    prompt_ids = encode_prompt(to_messages(ex), tok)
    return encode_completion(prompt_ids, ex["response"], tok, max_len)


def load_splits(
    tok: AutoTokenizer, data_dirs: dict[str, str], max_len: int = 512
) -> DatasetDict:
    """Read {train,val,test}.jsonl and tokenize every row.

    Returns a DatasetDict ("train", "val", "test") whose rows only have
    "input_ids" and "loss_mask" (original columns dropped).
    """
    dd = load_dataset(
        "json",
        data_files=data_dirs,
    )

    return dd.map(
        tokenize,
        fn_kwargs={"tok": tok, "max_len": max_len},
        remove_columns=dd["train"].column_names,  # drops prompt_id, prompt, response
    )


def collate(
    batch: list[dict], pad_id: int, left: bool = False
) -> dict[str, torch.Tensor]:
    """Left-pad rows to the longest one in the batch.

    params:
        batch: contains input_ids and loss_mask
        pad_id: what id should be used to create the pad
        left: bool to determine if we should use left pad

    Returns:
        long tensors of shape (B, T): "input_ids" (padded with pad_id),
        "attention_mask" and "loss_mask" (both 0 on padding).
    """
    batch_max = max(len(r["input_ids"]) for r in batch)
    ids, attn, loss = [], [], []
    for r in batch:
        n = batch_max - len(r["input_ids"])
        pad = lambda x, v: ([v] * n + x) if left else (x + [v] * n)

        # pad each llsit
        ids.append(pad(r["input_ids"], pad_id))
        attn.append(pad([1] * len(r["input_ids"]), 0))
        loss.append(pad(r["loss_mask"], 0))

    return {
        k: torch.tensor(v, dtype=torch.long)
        for k, v in [
            ("input_ids", ids),
            ("attention_mask", attn),
            ("loss_mask", loss),
        ]
    }
