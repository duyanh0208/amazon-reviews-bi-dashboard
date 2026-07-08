"""Central configuration for the BI & PA Amazon-reviews dashboard.

Keeping all tunables in one place follows the book's emphasis (Sharda, Ch.4)
on a well-governed data pipeline feeding the dashboard.
"""
from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
HF_CACHE = DATA / "hf_cache"
for _p in (RAW, PROCESSED, HF_CACHE):
    _p.mkdir(parents=True, exist_ok=True)

REVIEWS_PARQUET = PROCESSED / "reviews.parquet"          # cleaned + sentiment
EMOTIONS_PARQUET = PROCESSED / "reviews_emotions.parquet"  # GoEmotions scores

# --- Data source -------------------------------------------------------------
HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"
# (category -> friendly label). Picked for small download size + variety.
CATEGORIES = {
    "Digital_Music": "Digital Music",
    "Gift_Cards": "Gift Cards",
    "Magazine_Subscriptions": "Magazine Subscriptions",
    "Subscription_Boxes": "Subscription Boxes",
}
SAMPLE_PER_CATEGORY = 4000   # balanced sample to keep CPU scoring tractable
RANDOM_SEED = 42

# --- Models ------------------------------------------------------------------
# GoEmotions model mirrors the user's NLP project (same 28-label taxonomy).
EMOTION_MODEL = "SamLowe/roberta-base-go_emotions"
EMOTION_SAMPLE = 6000        # subset scored with the transformer (CPU-bound)
