import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import torch
    from datasets import DatasetDict
    from transformers import AutoTokenizer
    from pathlib import Path
    from datasets import load_dataset


    return AutoTokenizer, DatasetDict, Path, load_dataset, torch


@app.cell
def _(AutoTokenizer):
    name1 = "HuggingFaceTB/SmolLM-135M"
    name2 = "Qwen/Qwen2.5-0.5B"

    t1 = AutoTokenizer.from_pretrained(name1)
    t2 = AutoTokenizer.from_pretrained(name2)

    tokenizer = [t1, t2]
    return t1, tokenizer


@app.cell
def _():
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    return


@app.cell
def _(tokenizer):
    for t in tokenizer:
        print("special chars of", t.__class__.__name__)
        for t_id, t_name in zip(t.all_special_ids, t.all_special_tokens):
            print(t_id, t_name)
        print()
    return


@app.cell
def _(Path, t1):
    t1.chat_template = Path("../templates/qwen2-5_template.jinja").read_text()
    return


@app.cell
def _():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain gravity in one sentence."},
    ]
    return (messages,)


@app.cell
def _(messages, t1):
    formatted_prompt = t1.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # Set False during SFT training
    )
    print(formatted_prompt)
    return


@app.cell
def _(t1):
    t1.eot
    return


@app.cell
def _(AutoTokenizer):
    for name in ["HuggingFaceTB/SmolLM-135M", "Qwen/Qwen2.5-0.5B-Instruct"]:
        toki = AutoTokenizer.from_pretrained(name)
        print(name, "pad:", toki.pad_token, toki.pad_token_id, "| eos:", toki.eos_token, toki.eos_token_id)
    return


@app.cell
def _(AutoTokenizer, DatasetDict, Path, load_dataset, torch):
    SYSTEM = "You are a helpful assistant."

    def load_tokenizer(name: str):
        """Load a tokenizer with our ChatML chat template and a pad token set
        (falls back to "<|endoftext|>")."""
        tok = AutoTokenizer.from_pretrained(name)
        tok.chat_template = Path("../templates/qwen2-5_template.jinja").read_text()

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
            {"role": "user", "content": ex["prompt"]}
        ]


    def encode_prompt(messages: list[dict], tok: AutoTokenizer) -> list[int]:
        """Helper function to apply chat template.
        returns:
            Input ids of the chat template version of the messages
        """
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tok(text, add_special_tokens=False)["input_ids"]


    def encode_completion(prompt_ids: list[int], completion: str, tok: AutoTokenizer, max_len: int) -> dict:
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

    def load_splits(tok, data_dir: str = "data/dolly", max_len: int = 512) -> DatasetDict:
        """Read {train,val,test}.jsonl and tokenize every row.

        Returns a DatasetDict ("train", "val", "test") whose rows only have
        "input_ids" and "loss_mask" (original columns dropped).
        """
        dd = load_dataset(
            "json", 
            data_files={
                "train": "data/dolly/train.jsonl",
                "val":   "data/dolly/val.jsonl",
                "test":  "data/dolly/test.jsonl"
            }
        )

        return dd.map(
            tokenize,
            fn_kwargs={"tok": tok, "max_len": max_len},
            remove_columns=dd["train"].column_names,   # drops prompt_id, prompt, response
        )


    def collate(batch: list[dict], pad_id: int, left: bool = False) -> dict[str, torch.Tensor]:
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
            attn.append(pad(r["input_ids"], pad_id, 0))
            ids.append(pad(r["input_ids"], pad_id, 0))

        return {
            k: torch.Tensor(v, dtype=torch.long) 
                for k,v in [
                    ("input_ids", ids),
                    ("attention_mask", attn),
                    ("loss_mask", loss),
                ]
        }

    return (tokenize,)


@app.cell
def _(load_dataset):
    dd = load_dataset(
        "json", 
        data_files={
            "train": "../data/dolly/train.jsonl",
            "val":   "../data/dolly/val.jsonl",
            "test":  "../data/dolly/test.jsonl"
        }
    )
    return (dd,)


@app.cell
def _(dd, t1, tokenize):
    ds = dd.map(
        tokenize,
        fn_kwargs={"tok": t1, "max_len": 512},
        remove_columns=dd["train"].column_names,   # drops prompt_id, prompt, response
    )
    return (ds,)


@app.cell
def _(ds):
    ds
    return


@app.cell
def _(ds):
    ds["val"]["input_ids"]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
