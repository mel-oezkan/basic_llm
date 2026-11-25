import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import polars as pl


    return np, pl, plt


@app.cell
def _():
    from .config import MAX_TOKENS

    return


@app.cell
def _():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
    return (tok,)


@app.cell
def _(tok):
    tok
    return


@app.cell
def _(pl):
    df = pl.read_ndjson("../data/databricks-dolly-15k.jsonl")
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df, pl):
    instr_lens = df.select(pl.col("instruction").str.len_chars()).to_numpy().flatten()
    return (instr_lens,)


@app.cell
def _(np):
    def print_stats(arr):
        print(f"Mean: {np.mean(arr):.4f}")
        print(f"Std: {np.std(arr):.4f}")
        print(f"Median: {np.median(arr)}")
        print(f"Max: {np.max(arr)} - Min: {np.min(arr)}")

    return (print_stats,)


@app.cell
def _(instr_lens, print_stats):
    # mean, std and median of instruction lengths
    print_stats(instr_lens)
    return


@app.cell
def _(instr_lens, np, print_stats):
    # check how many samples are outsider 2*std
    outliers = instr_lens[(instr_lens > np.mean(instr_lens) + 2 * np.std(instr_lens))]
    n_outliers = np.sum(outliers)

    print_stats(outliers)
    return


@app.cell
def _(df, pl):
    filter_df = df.filter(
        (pl.col("instruction").str.len_chars() <= 420)
        & (pl.col("response").str.len_chars() > 30)
        & (pl.col("context").str.len_chars() == 0)
    )

    # drop the context column
    filter_df = filter_df.drop("context")
    filter_df
    return (filter_df,)


@app.cell
def _(df, pl):
    resp = (
        df.select(
            pl.col("category"),
            pl.col("response").str.len_chars().alias("response_len"),
        )
        .drop_nulls("category")
    )
    return (resp,)


@app.cell
def _(pl, plt, resp):
    avg = (
        resp.group_by("category")
        .agg(pl.col("response_len").mean())
        .sort("response_len")
    )
    plt.figure(figsize=(10, 5))
    plt.bar(avg["category"].to_list(), avg["response_len"].to_list())
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean response length (characters)")
    plt.title("Mean response length by category")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(filter_df):
    filter_df
    return


@app.cell
def _():
    from rapidfuzz import process, fuzz, utils

    return fuzz, process, utils


@app.cell
def _(fuzz, np, process, utils):
    def near_dedup(prompts: list[str], threshold: int = 90) -> list[int]:
        """Return indices to keep; the first occurrence of each near-duplicate group wins."""
        scores = process.cdist(
            prompts, prompts,
            scorer=fuzz.token_sort_ratio,      # word-order-insensitive, 0..100
            processor=utils.default_process,   # lowercases, strips punctuation/whitespace
            score_cutoff=threshold,            # anything below the threshold becomes 0
            dtype=np.uint8,
            workers=-1,
        )

        n = len(prompts)
        dup_of = np.full(n, -1)          # -1 = kept, otherwise index of the kept prompt it matched
        keep, pairs = [], []
        for i in range(n):
            if dup_of[i] != -1:
                continue
            keep.append(i)
            hits = np.where((scores[i, i + 1:] >= threshold) & (dup_of[i + 1:] == -1))[0] + i + 1
            dup_of[hits] = i
            pairs += [(i, j, int(scores[i, j])) for j in hits]
        return keep, pairs

    return (near_dedup,)


@app.cell
def _(filter_df, near_dedup):
    prompts =filter_df["instruction"].to_numpy()
    kept, pairs = near_dedup(prompts)
    return kept, pairs, prompts


@app.cell
def _(pairs, prompts):
    print_n = 10
    for i, j, s in sorted(pairs, key=lambda p: p[2])[:print_n]:   # lowest scores = closest to the threshold
        print(f"[{s}]  KEPT:    {prompts[i]!r}")
        print(f"       DROPPED: {prompts[j]!r}\n")
    return


@app.cell
def _(pairs, pl, prompts):
    pairs_df = pl.DataFrame(
        {
            "score":   [s for _, _, s in pairs],
            "kept":    [prompts[i] for i, _, _ in pairs],
            "dropped": [prompts[j] for _, j, _ in pairs],
        }
    ).sort("score")
    pairs_df   # marimo renders it as a sortable, searchable table
    return


@app.cell
def _(filter_df, kept):
    dedup_df = filter_df[kept]
    dedup_df
    return (dedup_df,)


@app.cell
def _(dedup_df, np, tok):

    # check how long the tokenized version of the dataset might be
    def token_lens(texts):
        ids = tok(list(texts), add_special_tokens=False)["input_ids"]
        return np.array([len(x) for x in ids])

    instr_tok_lens = token_lens(dedup_df["instruction"])
    resp_tok_lens = token_lens(dedup_df["response"])
    total_tok_lens = instr_tok_lens + resp_tok_lens

    return instr_tok_lens, resp_tok_lens, total_tok_lens


@app.cell
def _(instr_tok_lens, np, print_stats, resp_tok_lens, total_tok_lens):
    for name, arr in [("instruction", instr_tok_lens), ("response", resp_tok_lens), ("total", total_tok_lens)]:
        print(f"--- {name} ---")
        print_stats(arr)
        print(f"p95: {np.percentile(arr, 95):.0f}  p99: {np.percentile(arr, 99):.0f}\n")
    return


@app.cell
def _(dedup_df, pl, total_tok_lens):
    final_df = dedup_df.with_columns(pl.Series("total_token", total_tok_lens))
    final_df = final_df.filter(pl.col("total_token") <= 256)
    return (final_df,)


@app.cell
def _(final_df):
    final_df
    return


@app.cell
def _(EVAL_SIZE, TEST_SIZE, TRAIN_SIZE, dedup_df):
    TOTAL = TRAIN_SIZE + EVAL_SIZE + TEST_SIZE
    subset_n = dedup_df.sample(n=TOTAL, seed=42, shuffle=True)
    return (subset_n,)


@app.cell
def _(pl, subset_n):
    final_dataset = subset_n.select(
        pl.int_range(pl.len(), dtype=pl.UInt32).alias("prompt_id"),
        pl.col("instruction").alias("prompt"),
        pl.col("response"),
        pl.struct(pl.col("category")).alias("meta"),
        pl.lit("dolly-15k").alias("source_data")
    )
    return (final_dataset,)


@app.cell
def _(final_dataset):
    final_dataset
    return


@app.cell
def _(EVAL_SIZE, TRAIN_SIZE, final_dataset):
    splits = {
        "train": final_dataset[:TRAIN_SIZE],
        "val":   final_dataset[TRAIN_SIZE:TRAIN_SIZE + EVAL_SIZE],
        "test":  final_dataset[TRAIN_SIZE + EVAL_SIZE:],
    }

    for _name, split in splits.items():
        split.write_ndjson(f"../data/dolly/{_name}.jsonl")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
