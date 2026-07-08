"""
BI & PA - Amazon Reviews Emotion Analytics Dashboard (English UI)
Same data and logic as app.py; interface strings translated to English
for the report screenshots.

Run:  streamlit run app_en.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
REVIEWS = ROOT / "data" / "processed" / "reviews.parquet"
EMOTIONS = ROOT / "data" / "processed" / "reviews_emotions.parquet"

st.set_page_config(page_title="Amazon Reviews BI", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

SENT_COLORS = {"Positive": "#2E8B57", "Neutral": "#9AA0A6", "Negative": "#C0392B"}


@st.cache_data(show_spinner=False)
def load_reviews() -> pd.DataFrame:
    df = pd.read_parquet(REVIEWS)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_emotions() -> pd.DataFrame | None:
    if EMOTIONS.exists():
        return pd.read_parquet(EMOTIONS)
    return None


if not REVIEWS.exists():
    st.title("📊 Amazon Reviews BI Dashboard")
    st.warning("Data not ready. Run `python -m src.ingest` "
               "(and `python -m src.score_emotions` for the Emotion tab).")
    st.stop()

df_all = load_reviews()
emo_all = load_emotions()

# --------------------------------------------------------------------------- #
# Sidebar filters (guided analytics, Sharda Ch.4)                             #
# --------------------------------------------------------------------------- #
st.sidebar.header("🔎 Filters")
cats = sorted(df_all["category"].unique())
sel_cats = st.sidebar.multiselect("Category", cats, default=cats)

yr_min, yr_max = int(df_all["year"].min()), int(df_all["year"].max())
sel_years = st.sidebar.slider("Year range", yr_min, yr_max, (yr_min, yr_max))

sel_sent = st.sidebar.multiselect(
    "Sentiment", ["Positive", "Neutral", "Negative"],
    default=["Positive", "Neutral", "Negative"])
only_verified = st.sidebar.checkbox("Verified purchases only", value=False)

mask = (
    df_all["category"].isin(sel_cats)
    & df_all["year"].between(*sel_years)
    & df_all["sentiment"].astype(str).isin(sel_sent)
)
if only_verified:
    mask &= df_all["verified_purchase"]
df = df_all[mask].copy()

st.sidebar.markdown(f"**{len(df):,}** / {len(df_all):,} reviews after filtering")
st.sidebar.caption("Source: Amazon Reviews 2023 (McAuley-Lab, HuggingFace). "
                   "Balanced sample by category.")

if df.empty:
    st.title("📊 Amazon Reviews BI Dashboard")
    st.error("No data matches the filters. Please widen the filters in the sidebar.")
    st.stop()

# --------------------------------------------------------------------------- #
# Header                                                                       #
# --------------------------------------------------------------------------- #
st.title("📊 Amazon Reviews — Business Intelligence Dashboard")
st.caption("Descriptive · sentiment · emotion · predictive · prescriptive — "
           "BI & PA project (Sharda framework).")

tab_over, tab_cat, tab_text, tab_emo, tab_pred, tab_presc = st.tabs(
    ["① Overview", "② Category & Product", "③ Sentiment & Text",
     "④ Emotion (GoEmotions)", "⑤ Prediction", "⑥ Recommended Actions"])

# --------------------------------------------------------------------------- #
# TAB 1 - Overview                                                            #
# --------------------------------------------------------------------------- #
with tab_over:
    c = st.columns(5)
    c[0].metric("Total reviews", f"{len(df):,}")
    c[1].metric("Mean rating", f"{df['rating'].mean():.2f} ★")
    pos_pct = (df["sentiment"].astype(str).eq("Positive").mean()) * 100
    c[2].metric("% positive", f"{pos_pct:.1f}%")
    c[3].metric("Products", f"{df['parent_asin'].nunique():,}")
    c[4].metric("Mean helpful votes", f"{df['helpful_vote'].mean():.2f}")

    st.divider()
    a, b = st.columns(2)
    with a:
        st.subheader("Rating distribution")
        rc = df["rating"].value_counts().sort_index().reset_index()
        rc.columns = ["rating", "count"]
        fig = px.bar(rc, x="rating", y="count", text="count",
                     color="rating", color_continuous_scale="RdYlGn")
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Star rating",
                          yaxis_title="Number of reviews")
        st.plotly_chart(fig, width="stretch")
    with b:
        st.subheader("Sentiment composition")
        sc = df["sentiment"].astype(str).value_counts().reset_index()
        sc.columns = ["sentiment", "count"]
        fig = px.pie(sc, names="sentiment", values="count", hole=0.5,
                     color="sentiment", color_discrete_map=SENT_COLORS)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Review volume over time")
    ts = (df.groupby("year_month")
            .agg(reviews=("review_id", "count"),
                 avg_rating=("rating", "mean")).reset_index())
    ts["year_month"] = pd.to_datetime(ts["year_month"])
    ts = ts.sort_values("year_month")
    fig = px.area(ts, x="year_month", y="reviews")
    fig.update_traces(line_color="#4C78A8")
    fig.update_layout(xaxis_title="Time", yaxis_title="Number of reviews")
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
# TAB 2 - Category & Product                                                  #
# --------------------------------------------------------------------------- #
with tab_cat:
    a, b = st.columns(2)
    with a:
        st.subheader("Mean rating by category")
        cc = (df.groupby("category")
                .agg(avg=("rating", "mean"), n=("review_id", "count"))
                .reset_index().sort_values("avg"))
        fig = px.bar(cc, x="avg", y="category", orientation="h", text="n",
                     color="avg", color_continuous_scale="RdYlGn",
                     range_color=(1, 5))
        fig.update_traces(texttemplate="n=%{text}")
        fig.update_layout(coloraxis_showscale=False,
                          xaxis_title="Mean rating", yaxis_title="")
        st.plotly_chart(fig, width="stretch")
    with b:
        st.subheader("Sentiment mix by category")
        sm = (df.assign(s=df["sentiment"].astype(str))
                .groupby(["category", "s"]).size().reset_index(name="n"))
        fig = px.bar(sm, x="category", y="n", color="s", barmode="stack",
                     color_discrete_map=SENT_COLORS)
        fig.update_layout(xaxis_title="", yaxis_title="Number of reviews",
                          legend_title="Sentiment")
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Top / Bottom products by rating")
    min_n = st.slider("Minimum reviews per product", 3, 50, 10)
    name_col = "product_title" if df["product_title"].notna().any() else "parent_asin"
    prod = (df.groupby(["parent_asin"])
              .agg(name=(name_col, "first"),
                   category=("category", "first"),
                   avg_rating=("rating", "mean"),
                   reviews=("review_id", "count"),
                   sentiment=("sentiment_score", "mean")).reset_index())
    prod = prod[prod["reviews"] >= min_n]
    if prod.empty:
        st.info("No product meets the review threshold. Lower it above.")
    else:
        prod["name"] = prod["name"].fillna(prod["parent_asin"]).astype(str).str.slice(0, 50)
        t1, t2 = st.columns(2)
        with t1:
            st.caption("⭐ Best")
            st.dataframe(prod.sort_values("avg_rating", ascending=False)
                         .head(10)[["name", "category", "avg_rating", "reviews"]]
                         .round(2), hide_index=True, width="stretch")
        with t2:
            st.caption("⚠️ Worst")
            st.dataframe(prod.sort_values("avg_rating")
                         .head(10)[["name", "category", "avg_rating", "reviews"]]
                         .round(2), hide_index=True, width="stretch")

    st.divider()
    st.subheader("Helpful votes vs rating")
    hv = (df.groupby("rating").agg(helpful=("helpful_vote", "mean"),
                                   n=("review_id", "count")).reset_index())
    fig = px.bar(hv, x="rating", y="helpful", text=hv["n"],
                 color="rating", color_continuous_scale="RdYlGn")
    fig.update_traces(texttemplate="n=%{text}")
    fig.update_layout(coloraxis_showscale=False, xaxis_title="Star rating",
                      yaxis_title="Mean helpful votes")
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
# TAB 3 - Sentiment & Text                                                    #
# --------------------------------------------------------------------------- #
with tab_text:
    a, b = st.columns(2)
    with a:
        st.subheader("Sentiment vs star rating (cross-check)")
        ct = (df.assign(s=df["sentiment"].astype(str))
                .groupby(["rating", "s"]).size().reset_index(name="n"))
        piv = ct.pivot(index="rating", columns="s", values="n").fillna(0)
        fig = px.imshow(piv, text_auto=True, aspect="auto",
                        color_continuous_scale="Blues",
                        labels=dict(color="Reviews"))
        fig.update_layout(xaxis_title="Sentiment (VADER)", yaxis_title="Star rating")
        st.plotly_chart(fig, width="stretch")
    with b:
        st.subheader("Distribution of sentiment scores")
        fig = px.histogram(df, x="sentiment_score", nbins=40,
                           color=df["sentiment"].astype(str),
                           color_discrete_map=SENT_COLORS)
        fig.update_layout(xaxis_title="VADER compound", yaxis_title="Number of reviews",
                          legend_title="Sentiment")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Sentiment trend over time")
    tr = (df.assign(s=df["sentiment"].astype(str))
            .groupby(["year_month", "s"]).size().reset_index(name="n"))
    tot = tr.groupby("year_month")["n"].transform("sum")
    tr["share"] = tr["n"] / tot
    tr["year_month"] = pd.to_datetime(tr["year_month"])
    tr = tr.sort_values("year_month")
    fig = px.area(tr, x="year_month", y="share", color="s",
                  color_discrete_map=SENT_COLORS, groupnorm="fraction")
    fig.update_layout(xaxis_title="Time", yaxis_title="Share",
                      legend_title="Sentiment")
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Sample reviews")
    pick = st.radio("Filter by sentiment", ["Negative", "Neutral", "Positive"],
                    horizontal=True, index=0)
    sub = df[df["sentiment"].astype(str) == pick].nlargest(8, "helpful_vote")
    for _, r in sub.iterrows():
        st.markdown(f"**{'★'*int(r['rating'])}{'☆'*(5-int(r['rating']))}** · "
                    f"_{r['category']}_ · 👍 {r['helpful_vote']} · "
                    f"score={r['sentiment_score']:.2f}")
        st.write((r["title"] + " — " if r["title"] else "") + r["text"][:400])
        st.divider()

# --------------------------------------------------------------------------- #
# TAB 4 - Emotion (GoEmotions)                                                #
# --------------------------------------------------------------------------- #
with tab_emo:
    if emo_all is None:
        st.info("No emotion data yet. Run `python -m src.score_emotions` "
                "to assign the 28 GoEmotions labels to the review sample.")
    else:
        emo = emo_all[emo_all["category"].isin(sel_cats)].copy()
        st.caption(f"GoEmotions model (28 emotions) on {len(emo):,} reviews, "
                   "the same taxonomy as the NLP project.")
        a, b = st.columns([2, 3])
        with a:
            st.subheader("Top emotions")
            vc = emo["top_emotion"].value_counts().head(12).reset_index()
            vc.columns = ["emotion", "n"]
            fig = px.bar(vc, x="n", y="emotion", orientation="h",
                         color="n", color_continuous_scale="Viridis")
            fig.update_layout(coloraxis_showscale=False, yaxis_title="",
                              xaxis_title="Number of reviews")
            fig.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig, width="stretch")
        with b:
            st.subheader("Emotion × category (share)")
            top_em = emo["top_emotion"].value_counts().head(8).index
            hm = (emo[emo["top_emotion"].isin(top_em)]
                  .groupby(["category", "top_emotion"]).size().reset_index(name="n"))
            piv = hm.pivot(index="top_emotion", columns="category", values="n").fillna(0)
            piv = piv.div(piv.sum(axis=0), axis=1)
            fig = px.imshow(piv, text_auto=".0%", aspect="auto",
                            color_continuous_scale="Purples")
            fig.update_layout(xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, width="stretch")

        st.subheader("Mean emotion intensity by star rating")
        merged = emo.merge(df[["review_id", "rating"]], on="review_id", how="inner")
        emo_cols = [c for c in emo.columns if c.startswith("emo_")]
        show = ["emo_admiration", "emo_joy", "emo_gratitude", "emo_disappointment",
                "emo_anger", "emo_annoyance", "emo_disgust", "emo_sadness"]
        show = [c for c in show if c in emo_cols]
        g = merged.groupby("rating")[show].mean().reset_index()
        gm = g.melt(id_vars="rating", var_name="emotion", value_name="score")
        gm["emotion"] = gm["emotion"].str.replace("emo_", "")
        fig = px.line(gm, x="rating", y="score", color="emotion", markers=True)
        fig.update_layout(xaxis_title="Star rating", yaxis_title="Mean probability")
        st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
# TAB 5 - Predictive                                                          #
# --------------------------------------------------------------------------- #
with tab_pred:
    st.subheader("Predicting negative reviews (rating ≤ 2)")
    st.caption("Logistic Regression, illustrating the data-mining process "
               "(business understanding → data prep → model → evaluation).")
    if len(df) < 200 or df["rating"].le(2).sum() < 20:
        st.info("More data is needed (widen the filters) to train a stable model.")
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (roc_auc_score, confusion_matrix,
                                      classification_report)

        feats = ["sentiment_score", "word_count", "helpful_vote"]
        d = df.copy()
        d["verified"] = d["verified_purchase"].astype(int)
        feats.append("verified")
        X = d[feats].fillna(0).values
        y = d["rating"].le(2).astype(int).values
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(yte, proba)

        m = st.columns(3)
        m[0].metric("ROC-AUC", f"{auc:.3f}")
        m[1].metric("Negative rate", f"{y.mean()*100:.1f}%")
        m[2].metric("Test samples", f"{len(yte):,}")

        a, b = st.columns(2)
        with a:
            st.caption("Confusion matrix")
            cm = confusion_matrix(yte, pred)
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                            x=["Pred: Not neg.", "Pred: Negative"],
                            y=["True: Not neg.", "True: Negative"])
            st.plotly_chart(fig, width="stretch")
        with b:
            st.caption("Feature weights (effect)")
            coef = pd.DataFrame({"feature": feats, "coef": clf.coef_[0]})
            coef = coef.sort_values("coef")
            fig = px.bar(coef, x="coef", y="feature", orientation="h",
                         color="coef", color_continuous_scale="RdBu")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")

        with st.expander("Detailed classification report"):
            rep = classification_report(yte, pred, output_dict=True,
                                        target_names=["Not neg.", "Negative"])
            st.dataframe(pd.DataFrame(rep).T.round(3))

# --------------------------------------------------------------------------- #
# TAB 6 - Prescriptive                                                        #
# --------------------------------------------------------------------------- #
with tab_presc:
    st.subheader("Recommended actions — which products to prioritise?")
    st.caption("Prescriptive analytics: turning insight into decisions — "
               "risk scoring, a decision matrix and constrained resource optimisation.")

    min_np = st.slider("Minimum reviews per product", 3, 50, 8, key="presc_min")
    name_col = "product_title" if df["product_title"].notna().any() else "parent_asin"
    d = df.copy()
    d["is_neg"] = d["sentiment"].astype(str).eq("Negative")

    prod = (d.groupby("parent_asin")
              .agg(name=(name_col, "first"),
                   category=("category", "first"),
                   reviews=("review_id", "count"),
                   avg_rating=("rating", "mean"),
                   neg_count=("is_neg", "sum"),
                   helpful=("helpful_vote", "mean")).reset_index())
    prod = prod[prod["reviews"] >= min_np].copy()

    if prod.empty:
        st.info("No product meets the review threshold. Lower it above.")
    else:
        prod["neg_share"] = prod["neg_count"] / prod["reviews"]
        prod["name"] = prod["name"].fillna(prod["parent_asin"]).astype(str).str.slice(0, 45)
        vol_med = prod["reviews"].median()

        def action_tier(r):
            if r["avg_rating"] < 3.0 and r["reviews"] >= vol_med:
                return "🔴 Fix now"
            if r["avg_rating"] < 3.5:
                return "🟡 Improve / monitor"
            if r["avg_rating"] >= 4.0 and r["reviews"] >= vol_med:
                return "🟢 Leverage / promote"
            return "⚪ Maintain"

        prod["action"] = prod.apply(action_tier, axis=1)
        TIER_COLORS = {"🔴 Fix now": "#C0392B", "🟡 Improve / monitor": "#E1A100",
                       "🟢 Leverage / promote": "#2E8B57", "⚪ Maintain": "#9AA0A6"}

        k1, k2, k3 = st.columns(3)
        urgent = prod["action"].eq("🔴 Fix now").sum()
        k1.metric("Products to fix now", f"{urgent}")
        k2.metric("Negative reviews (filtered)", f"{int(prod['neg_count'].sum()):,}")
        k3.metric("Products assessed", f"{len(prod):,}")

        st.divider()
        st.subheader("Decision matrix: reach × satisfaction")
        fig = px.scatter(prod, x="reviews", y="avg_rating", size="neg_count",
                         color="action", color_discrete_map=TIER_COLORS,
                         hover_name="name", size_max=40,
                         labels={"reviews": "Number of reviews (reach)",
                                 "avg_rating": "Mean rating (satisfaction)"})
        fig.add_hline(y=3.5, line_dash="dot", line_color="grey")
        fig.add_vline(x=vol_med, line_dash="dot", line_color="grey")
        fig.update_layout(legend_title="Recommended action", yaxis_range=[1, 5])
        st.plotly_chart(fig, width="stretch")
        st.caption("Lower-right (many reviews, low rating) = highest impact, fix first. "
                   "Upper-right = products to promote.")

        st.divider()
        st.subheader("Resource optimisation: where to focus to cut the most complaints?")
        rank = prod.sort_values("neg_count", ascending=False).reset_index(drop=True)
        total_neg = int(rank["neg_count"].sum())
        max_k = int(len(rank))
        cap = st.slider("Resource: number of products handled this cycle", 1,
                        max_k, min(5, max_k), key="presc_cap")
        chosen = rank.head(cap)
        covered = int(chosen["neg_count"].sum())
        cov_pct = covered / total_neg * 100 if total_neg else 0

        c1, c2 = st.columns([2, 3])
        with c1:
            st.metric("% of negative reviews addressed", f"{cov_pct:.0f}%",
                      help=f"Handling {cap}/{max_k} products addresses "
                           f"{covered:,}/{total_neg:,} negative reviews.")
            st.caption(f"Focusing on just **{cap}/{max_k}** products "
                       f"addresses **{cov_pct:.0f}%** of complaints (Pareto principle).")
        with c2:
            rank["cum_pct"] = rank["neg_count"].cumsum() / total_neg * 100 if total_neg else 0
            rank["idx"] = range(1, len(rank) + 1)
            fig = px.area(rank, x="idx", y="cum_pct",
                          labels={"idx": "Prioritised products (descending)",
                                  "cum_pct": "Cumulative % of complaints"})
            fig.add_vline(x=cap, line_dash="dash", line_color="#C0392B")
            fig.update_traces(line_color="#4C78A8")
            fig.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig, width="stretch")

        st.markdown("**Prioritised action list (this cycle):**")
        st.dataframe(
            chosen[["name", "category", "reviews", "avg_rating",
                    "neg_count", "neg_share", "action"]]
            .rename(columns={"name": "Product", "category": "Category",
                             "reviews": "Reviews", "avg_rating": "Mean rating",
                             "neg_count": "Negative reviews", "neg_share": "Neg. share",
                             "action": "Action"}).round(2),
            hide_index=True, width="stretch")

        st.divider()
        st.subheader("Category-level recommendations")
        cat_sum = (d.groupby("category")
                     .agg(reviews=("review_id", "count"),
                          avg_rating=("rating", "mean"),
                          neg_share=("is_neg", "mean")).reset_index())
        for _, r in cat_sum.sort_values("neg_share", ascending=False).iterrows():
            if r["neg_share"] >= 0.15:
                msg = (f"⚠️ **{r['category']}** — high negative share "
                       f"({r['neg_share']*100:.0f}%). Review product quality and descriptions.")
            elif r["avg_rating"] >= 4.3:
                msg = (f"✅ **{r['category']}** — high satisfaction "
                       f"(rating {r['avg_rating']:.2f}). Push marketing and encourage reviews.")
            else:
                msg = (f"➡️ **{r['category']}** — stable (rating {r['avg_rating']:.2f}, "
                       f"negative {r['neg_share']*100:.0f}%). Maintain and monitor.")
            st.markdown(msg)

st.sidebar.divider()
with st.sidebar.expander("ℹ️ Knowledge framework (Sharda)"):
    st.markdown(
        "- **Ch.3/4** Preprocessing, data warehousing, information dashboards\n"
        "- **Ch.5** Data-mining classification (tab ⑤)\n"
        "- **Ch.6** Text & sentiment/emotion analytics (tabs ③④)\n"
        "- **Ch.8** Prescriptive — recommendations & optimisation (tab ⑥)\n"
        "- Data: Amazon Reviews 2023 · Emotion: GoEmotions (bridge to the NLP project)")
