"""Stage 1 — Data ingestion & preparation (Sharda Ch.3: data preprocessing).

Downloads a balanced sample of Amazon Reviews 2023 across a few categories,
cleans them, joins lightweight product metadata, computes a fast rule-based
sentiment (VADER), and writes a single tidy parquet for the dashboard.

Run:  python -m src.ingest
"""
from __future__ import annotations
import sys
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download, HfApi

from .config import (
    HF_REPO, CATEGORIES, SAMPLE_PER_CATEGORY, RANDOM_SEED,
    RAW, HF_CACHE, REVIEWS_PARQUET,
)

META_SIZE_LIMIT_MB = 40  # skip oversized meta files to keep the download light


def _download(filename: str) -> str:
    return hf_hub_download(
        repo_id=HF_REPO, filename=filename, repo_type="dataset",
        cache_dir=str(HF_CACHE), local_dir=str(RAW),
    )


def _file_sizes() -> dict[str, float]:
    info = HfApi().repo_info(HF_REPO, repo_type="dataset", files_metadata=True)
    return {s.rfilename: (s.size or 0) / 1e6 for s in info.siblings}


def load_category(cat: str, sizes: dict[str, float]) -> pd.DataFrame:
    print(f"[ingest] {cat}: downloading reviews ...", flush=True)
    rev_path = _download(f"raw/review_categories/{cat}.jsonl")
    df = pd.read_json(rev_path, lines=True)
    # balanced random sample
    n = min(SAMPLE_PER_CATEGORY, len(df))
    df = df.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)
    df["category"] = CATEGORIES[cat]

    # best-effort product metadata join (title / price / store)
    meta_file = f"raw/meta_categories/meta_{cat}.jsonl"
    if sizes.get(meta_file, 1e9) <= META_SIZE_LIMIT_MB:
        print(f"[ingest] {cat}: downloading metadata ...", flush=True)
        meta_path = _download(meta_file)
        meta = pd.read_json(meta_path, lines=True)
        meta = meta[["parent_asin", "title", "price", "store"]].rename(
            columns={"title": "product_title"})
        meta = meta.drop_duplicates("parent_asin")
        df = df.merge(meta, on="parent_asin", how="left")
    else:
        print(f"[ingest] {cat}: meta too large, skipping product join", flush=True)
        df["product_title"] = np.nan
        df["price"] = np.nan
        df["store"] = np.nan
    print(f"[ingest] {cat}: {len(df):,} rows", flush=True)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["category", "rating", "title", "text", "parent_asin", "asin",
            "product_title", "price", "store", "helpful_vote",
            "verified_purchase", "timestamp"]
    df = df[[c for c in keep if c in df.columns]].copy()

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[df["rating"].between(1, 5)].copy()

    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    df["helpful_vote"] = pd.to_numeric(df.get("helpful_vote"), errors="coerce").fillna(0).astype(int)
    df["verified_purchase"] = df.get("verified_purchase", False).fillna(False).astype(bool)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["review_len"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()

    df = df.reset_index(drop=True)
    df["review_id"] = df.index
    return df


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Fast rule-based polarity for the whole sample (Sharda Ch.6)."""
    import nltk
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon")
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()

    print("[ingest] scoring VADER sentiment ...", flush=True)
    scores = df["text"].map(lambda t: sia.polarity_scores(t)["compound"])
    df["sentiment_score"] = scores.astype(float)
    df["sentiment"] = pd.cut(
        df["sentiment_score"], bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["Negative", "Neutral", "Positive"])
    return df


def main() -> int:
    sizes = _file_sizes()
    frames = [load_category(cat, sizes) for cat in CATEGORIES]
    df = pd.concat(frames, ignore_index=True)
    df = clean(df)
    df = add_sentiment(df)
    df.to_parquet(REVIEWS_PARQUET, index=False)
    print(f"\n[ingest] DONE -> {REVIEWS_PARQUET}")
    print(f"[ingest] rows={len(df):,}  categories={df['category'].nunique()}  "
          f"date range={df['date'].min().date()}..{df['date'].max().date()}")
    print(df["sentiment"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
