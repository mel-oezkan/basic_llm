"""Build train/val/test splits from databricks-dolly-15k.

Pipeline: char-length filter -> near-duplicate removal -> token-length filter
-> seeded sample -> split. Writes data/dolly/{train,val,test}.jsonl and meta.json.

Run from anywhere:  python scripts/prepare_dolly.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process, utils
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import EVAL_SIZE, MAX_TOKENS, MODEL, TEST_SIZE, TRAIN_SIZE  # noqa: E402

RAW_PATH = ROOT / "data" / "databricks-dolly-15k.jsonl"
OUT_DIR = ROOT / "data" / "dolly"

MAX_INSTR_CHARS = 420
MIN_RESP_CHARS = 30
DEDUP_THRESHOLD = 90
SEED = 42


def near_dedup(
    prompts: list[str], threshold: int = DEDUP_THRESHOLD
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Return (indices to keep, (kept, dropped, score) pairs).

    The first occurrence of each near-duplicate group wins.
    """
    scores = process.cdist(
        prompts,
        prompts,
        scorer=fuzz.token_sort_ratio,  # word-order-insensitive, 0..100
        processor=utils.default_process,  # lowercases, strips punctuation/whitespace
        score_cutoff=threshold,  # anything below the threshold becomes 0
        dtype=np.uint8,
        workers=-1,
    )

    n = len(prompts)
    dup_of = np.full(n, -1)  # -1 = kept, otherwise index of the kept prompt it matched
    keep, pairs = [], []

    for i in range(n):
        if dup_of[i] != -1:
            continue
        keep.append(i)
        rest = slice(i + 1, None)
        hits = np.where((scores[i, rest] >= threshold) & (dup_of[rest] == -1))[0] + i + 1
        dup_of[hits] = i
        pairs += [(i, int(j), int(scores[i, j])) for j in hits]

    return keep, pairs


def token_lens(tokenizer, texts) -> np.ndarray:
    ids = tokenizer(list(texts), add_special_tokens=False)["input_ids"]
    return np.array([len(x) for x in ids])


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    counts = {}

    df = pl.read_ndjson(RAW_PATH)
    counts["raw"] = len(df)

    # 1. char-length filter; keep only closed-book samples (no context)
    df = df.filter(
        (pl.col("instruction").str.len_chars() <= MAX_INSTR_CHARS)
        & (pl.col("response").str.len_chars() > MIN_RESP_CHARS)
        & (pl.col("context").str.len_chars() == 0)
    ).drop("context")
    counts["filtered"] = len(df)

    # 2. near-duplicate removal on the instruction
    kept, pairs = near_dedup(df["instruction"].to_list())
    df = df[kept]
    counts["dedup"] = len(df)
    print(f"near-dedup dropped {len(pairs)} prompts")

    # 3. token-length filter (instruction + response, without chat template)
    total_tokens = token_lens(tok, df["instruction"]) + token_lens(tok, df["response"])
    df = df.with_columns(pl.Series("total_tokens", total_tokens)).filter(
        pl.col("total_tokens") <= MAX_TOKENS
    )
    counts["token_filtered"] = len(df)

    # 4. sample and reshape into the shared prompt schema
    total = TRAIN_SIZE + EVAL_SIZE + TEST_SIZE
    if len(df) < total:
        raise ValueError(f"only {len(df)} rows left, need {total}")

    subset = df.sample(n=total, seed=SEED, shuffle=True)
    dataset = subset.select(
        pl.int_range(pl.len(), dtype=pl.UInt32).alias("prompt_id"),
        pl.col("instruction").alias("prompt"),
        pl.col("response"),
        pl.struct(pl.col("category")).alias("meta"),
        pl.lit("dolly-15k").alias("source_data"),
    )

    # 5. split and write
    splits = {
        "train": dataset[:TRAIN_SIZE],
        "val": dataset[TRAIN_SIZE : TRAIN_SIZE + EVAL_SIZE],
        "test": dataset[TRAIN_SIZE + EVAL_SIZE :],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.write_ndjson(OUT_DIR / f"{name}.jsonl")

    meta = {
        "source": "databricks-dolly-15k",
        "count_tokenizer": MODEL,
        "max_total_tokens": MAX_TOKENS,
        "observed_max_total_tokens": int(subset["total_tokens"].to_numpy().max()),
        "filters": {
            "max_instr_chars": MAX_INSTR_CHARS,
            "min_resp_chars": MIN_RESP_CHARS,
            "context": "empty only",
        },
        "dedup": {"scorer": "token_sort_ratio", "threshold": DEDUP_THRESHOLD},
        "seed": SEED,
        "rows_per_stage": counts,
        "sizes": {name: len(split) for name, split in splits.items()},
    }
    
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    for stage, n in counts.items():
        print(f"{stage:>15}: {n}")
    print(f"wrote {', '.join(splits)} + meta.json to {OUT_DIR}")


if __name__ == "__main__":
    main()
