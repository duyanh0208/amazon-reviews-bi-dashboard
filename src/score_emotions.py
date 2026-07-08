"""Stage 2 — Fine-grained emotion classification (predictive / text analytics).

Applies a GoEmotions model (28 emotions) to a stratified subset of reviews.
This is the bridge to the NLP course project, which studied exactly this
GoEmotions taxonomy. Runs on CPU; subset size is capped in config.

Run:  python -m src.score_emotions
"""
from __future__ import annotations
import sys
import pandas as pd

from .config import (
    REVIEWS_PARQUET, EMOTIONS_PARQUET, EMOTION_MODEL, EMOTION_SAMPLE,
    RANDOM_SEED,
)


def stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    frac = n / len(df)
    out = (df.groupby("category", group_keys=False)[df.columns.tolist()]
             .apply(lambda g: g.sample(frac=frac, random_state=RANDOM_SEED)))
    return out.reset_index(drop=True)


def main() -> int:
    if not REVIEWS_PARQUET.exists():
        print("[emotions] reviews.parquet missing — run src.ingest first")
        return 1

    df = pd.read_parquet(REVIEWS_PARQUET)
    sample = stratified_sample(df, EMOTION_SAMPLE)
    texts = sample["text"].str.slice(0, 1000).tolist()  # guard very long reviews
    print(f"[emotions] scoring {len(texts):,} reviews with {EMOTION_MODEL}", flush=True)

    from transformers import pipeline
    clf = pipeline("text-classification", model=EMOTION_MODEL,
                   top_k=None, truncation=True, max_length=128, device=-1)

    rows = []
    B = 32
    for i in range(0, len(texts), B):
        batch = clf(texts[i:i + B])
        for preds in batch:
            d = {p["label"]: float(p["score"]) for p in preds}
            top = max(preds, key=lambda p: p["score"])
            d["top_emotion"] = top["label"]
            d["top_score"] = float(top["score"])
            rows.append(d)
        if i % (B * 20) == 0:
            print(f"[emotions]  {i+len(batch):,}/{len(texts):,}", flush=True)

    emo = pd.DataFrame(rows)
    emo.columns = [c if c in ("top_emotion", "top_score") else f"emo_{c}"
                   for c in emo.columns]
    emo.insert(0, "review_id", sample["review_id"].values)
    emo.insert(1, "category", sample["category"].values)
    emo.to_parquet(EMOTIONS_PARQUET, index=False)
    print(f"\n[emotions] DONE -> {EMOTIONS_PARQUET}  ({len(emo):,} rows)")
    print(emo["top_emotion"].value_counts().head(12).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
