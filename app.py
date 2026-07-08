"""
BI & PA — Amazon Reviews Emotion Analytics Dashboard
=====================================================
Maps directly to Sharda, *Business Intelligence, Analytics, and Data Science*:
  - Ch.3/4  Descriptive analytics, data warehousing & information dashboards
  - Ch.5    Predictive analytics (data-mining classification)
  - Ch.6    Text / sentiment / emotion analytics

Run:  streamlit run app.py
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

PALETTE = px.colors.qualitative.Safe
SENT_COLORS = {"Positive": "#2E8B57", "Neutral": "#9AA0A6", "Negative": "#C0392B"}


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #
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
    st.warning("Dữ liệu chưa sẵn sàng. Hãy chạy:  `python -m src.ingest`  "
               "(và `python -m src.score_emotions` cho tab Cảm xúc).")
    st.stop()

df_all = load_reviews()
emo_all = load_emotions()

# --------------------------------------------------------------------------- #
# Sidebar filters (guided analytics — Sharda Ch.4)                            #
# --------------------------------------------------------------------------- #
st.sidebar.header("🔎 Bộ lọc")
cats = sorted(df_all["category"].unique())
sel_cats = st.sidebar.multiselect("Danh mục", cats, default=cats)

dmin, dmax = df_all["date"].min().date(), df_all["date"].max().date()
yr_min, yr_max = int(df_all["year"].min()), int(df_all["year"].max())
sel_years = st.sidebar.slider("Khoảng năm", yr_min, yr_max, (yr_min, yr_max))

sel_sent = st.sidebar.multiselect(
    "Sắc thái", ["Positive", "Neutral", "Negative"],
    default=["Positive", "Neutral", "Negative"])
only_verified = st.sidebar.checkbox("Chỉ đơn đã xác minh (verified)", value=False)

mask = (
    df_all["category"].isin(sel_cats)
    & df_all["year"].between(*sel_years)
    & df_all["sentiment"].astype(str).isin(sel_sent)
)
if only_verified:
    mask &= df_all["verified_purchase"]
df = df_all[mask].copy()

st.sidebar.markdown(f"**{len(df):,}** / {len(df_all):,} reviews sau lọc")
st.sidebar.caption("Nguồn: Amazon Reviews 2023 (McAuley-Lab, HuggingFace). "
                   "Mẫu cân bằng theo danh mục.")

if df.empty:
    st.title("📊 Amazon Reviews BI Dashboard")
    st.error("Không có dữ liệu khớp bộ lọc. Hãy nới bộ lọc ở thanh bên.")
    st.stop()


# --------------------------------------------------------------------------- #
# Header                                                                       #
# --------------------------------------------------------------------------- #
st.title("📊 Amazon Reviews — Business Intelligence Dashboard")
st.caption("Phân tích mô tả · sắc thái · cảm xúc · dự báo — Đồ án BI & PA "
           "(khung kiến thức Sharda).")

tab_over, tab_cat, tab_text, tab_emo, tab_pred, tab_presc = st.tabs(
    ["① Tổng quan", "② Danh mục & Sản phẩm", "③ Sắc thái & Văn bản",
     "④ Cảm xúc (GoEmotions)", "⑤ Dự báo", "⑥ Đề xuất hành động"])


# --------------------------------------------------------------------------- #
# TAB 1 — Overview (Level 1: glanceable KPIs)                                 #
# --------------------------------------------------------------------------- #
with tab_over:
    c = st.columns(5)
    c[0].metric("Tổng reviews", f"{len(df):,}")
    c[1].metric("Rating trung bình", f"{df['rating'].mean():.2f} ★")
    pos_pct = (df["sentiment"].astype(str).eq("Positive").mean()) * 100
    c[2].metric("% tích cực", f"{pos_pct:.1f}%")
    c[3].metric("Sản phẩm", f"{df['parent_asin'].nunique():,}")
    c[4].metric("Helpful votes TB", f"{df['helpful_vote'].mean():.2f}")

    st.divider()
    a, b = st.columns(2)
    with a:
        st.subheader("Phân bố rating")
        rc = df["rating"].value_counts().sort_index().reset_index()
        rc.columns = ["rating", "count"]
        fig = px.bar(rc, x="rating", y="count", text="count",
                     color="rating", color_continuous_scale="RdYlGn")
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Số sao",
                          yaxis_title="Số review")
        st.plotly_chart(fig, width="stretch")
    with b:
        st.subheader("Cơ cấu sắc thái")
        sc = df["sentiment"].astype(str).value_counts().reset_index()
        sc.columns = ["sentiment", "count"]
        fig = px.pie(sc, names="sentiment", values="count", hole=0.5,
                     color="sentiment", color_discrete_map=SENT_COLORS)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Khối lượng review theo thời gian")
    ts = (df.groupby("year_month")
            .agg(reviews=("review_id", "count"),
                 avg_rating=("rating", "mean")).reset_index())
    ts["year_month"] = pd.to_datetime(ts["year_month"])
    ts = ts.sort_values("year_month")
    fig = px.area(ts, x="year_month", y="reviews")
    fig.update_traces(line_color="#4C78A8")
    fig.update_layout(xaxis_title="Thời gian", yaxis_title="Số review")
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# TAB 2 — Category & Product (Level 2: comparison / trends)                   #
# --------------------------------------------------------------------------- #
with tab_cat:
    a, b = st.columns(2)
    with a:
        st.subheader("Rating trung bình theo danh mục")
        cc = (df.groupby("category")
                .agg(avg=("rating", "mean"), n=("review_id", "count"))
                .reset_index().sort_values("avg"))
        fig = px.bar(cc, x="avg", y="category", orientation="h", text="n",
                     color="avg", color_continuous_scale="RdYlGn",
                     range_color=(1, 5))
        fig.update_traces(texttemplate="n=%{text}")
        fig.update_layout(coloraxis_showscale=False,
                          xaxis_title="Rating TB", yaxis_title="")
        st.plotly_chart(fig, width="stretch")
    with b:
        st.subheader("Cơ cấu sắc thái theo danh mục")
        sm = (df.assign(s=df["sentiment"].astype(str))
                .groupby(["category", "s"]).size().reset_index(name="n"))
        fig = px.bar(sm, x="category", y="n", color="s", barmode="stack",
                     color_discrete_map=SENT_COLORS)
        fig.update_layout(xaxis_title="", yaxis_title="Số review",
                          legend_title="Sắc thái")
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Top / Bottom sản phẩm theo rating")
    min_n = st.slider("Số review tối thiểu / sản phẩm", 3, 50, 10)
    name_col = "product_title" if df["product_title"].notna().any() else "parent_asin"
    prod = (df.groupby(["parent_asin"])
              .agg(name=(name_col, "first"),
                   category=("category", "first"),
                   avg_rating=("rating", "mean"),
                   reviews=("review_id", "count"),
                   sentiment=("sentiment_score", "mean")).reset_index())
    prod = prod[prod["reviews"] >= min_n]
    if prod.empty:
        st.info("Không có sản phẩm nào đạt ngưỡng số review. Giảm ngưỡng ở trên.")
    else:
        prod["name"] = prod["name"].fillna(prod["parent_asin"]).astype(str).str.slice(0, 50)
        t1, t2 = st.columns(2)
        with t1:
            st.caption("⭐ Tốt nhất")
            st.dataframe(prod.sort_values("avg_rating", ascending=False)
                         .head(10)[["name", "category", "avg_rating", "reviews"]]
                         .round(2), hide_index=True, width="stretch")
        with t2:
            st.caption("⚠️ Kém nhất")
            st.dataframe(prod.sort_values("avg_rating")
                         .head(10)[["name", "category", "avg_rating", "reviews"]]
                         .round(2), hide_index=True, width="stretch")

    st.divider()
    st.subheader("Độ hữu ích (helpful votes) vs Rating")
    hv = (df.groupby("rating").agg(helpful=("helpful_vote", "mean"),
                                   n=("review_id", "count")).reset_index())
    fig = px.bar(hv, x="rating", y="helpful", text=hv["n"],
                 color="rating", color_continuous_scale="RdYlGn")
    fig.update_traces(texttemplate="n=%{text}")
    fig.update_layout(coloraxis_showscale=False, xaxis_title="Số sao",
                      yaxis_title="Helpful votes TB")
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# TAB 3 — Sentiment & Text                                                    #
# --------------------------------------------------------------------------- #
with tab_text:
    a, b = st.columns(2)
    with a:
        st.subheader("Sắc thái vs Số sao (kiểm chứng chéo)")
        ct = (df.assign(s=df["sentiment"].astype(str))
                .groupby(["rating", "s"]).size().reset_index(name="n"))
        piv = ct.pivot(index="rating", columns="s", values="n").fillna(0)
        fig = px.imshow(piv, text_auto=True, aspect="auto",
                        color_continuous_scale="Blues",
                        labels=dict(color="Số review"))
        fig.update_layout(xaxis_title="Sắc thái (VADER)", yaxis_title="Số sao")
        st.plotly_chart(fig, width="stretch")
    with b:
        st.subheader("Phân bố điểm sentiment")
        fig = px.histogram(df, x="sentiment_score", nbins=40,
                           color=df["sentiment"].astype(str),
                           color_discrete_map=SENT_COLORS)
        fig.update_layout(xaxis_title="VADER compound", yaxis_title="Số review",
                          legend_title="Sắc thái")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Xu hướng sắc thái theo thời gian")
    tr = (df.assign(s=df["sentiment"].astype(str))
            .groupby(["year_month", "s"]).size().reset_index(name="n"))
    tot = tr.groupby("year_month")["n"].transform("sum")
    tr["share"] = tr["n"] / tot
    tr["year_month"] = pd.to_datetime(tr["year_month"])
    tr = tr.sort_values("year_month")
    fig = px.area(tr, x="year_month", y="share", color="s",
                  color_discrete_map=SENT_COLORS, groupnorm="fraction")
    fig.update_layout(xaxis_title="Thời gian", yaxis_title="Tỷ trọng",
                      legend_title="Sắc thái")
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Đọc thử review")
    pick = st.radio("Lọc theo sắc thái", ["Negative", "Neutral", "Positive"],
                    horizontal=True, index=0)
    sub = df[df["sentiment"].astype(str) == pick].nlargest(8, "helpful_vote")
    for _, r in sub.iterrows():
        st.markdown(f"**{'★'*int(r['rating'])}{'☆'*(5-int(r['rating']))}** · "
                    f"_{r['category']}_ · 👍 {r['helpful_vote']} · "
                    f"score={r['sentiment_score']:.2f}")
        st.write((r["title"] + " — " if r["title"] else "") + r["text"][:400])
        st.divider()


# --------------------------------------------------------------------------- #
# TAB 4 — Emotion (GoEmotions, Sharda Ch.6 + NLP project bridge)              #
# --------------------------------------------------------------------------- #
with tab_emo:
    if emo_all is None:
        st.info("Chưa có dữ liệu cảm xúc. Chạy:  `python -m src.score_emotions`  "
                "để gán 28 nhãn GoEmotions cho mẫu review.")
    else:
        emo = emo_all[emo_all["category"].isin(sel_cats)].copy()
        st.caption(f"Mô hình GoEmotions (28 cảm xúc) trên {len(emo):,} review — "
                   "cùng taxonomy với đồ án NLP.")
        a, b = st.columns([2, 3])
        with a:
            st.subheader("Top cảm xúc")
            vc = emo["top_emotion"].value_counts().head(12).reset_index()
            vc.columns = ["emotion", "n"]
            fig = px.bar(vc, x="n", y="emotion", orientation="h",
                         color="n", color_continuous_scale="Viridis")
            fig.update_layout(coloraxis_showscale=False, yaxis_title="",
                              xaxis_title="Số review")
            fig.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig, width="stretch")
        with b:
            st.subheader("Cảm xúc × Danh mục (tỷ trọng)")
            top_em = emo["top_emotion"].value_counts().head(8).index
            hm = (emo[emo["top_emotion"].isin(top_em)]
                  .groupby(["category", "top_emotion"]).size().reset_index(name="n"))
            piv = hm.pivot(index="top_emotion", columns="category", values="n").fillna(0)
            piv = piv.div(piv.sum(axis=0), axis=1)
            fig = px.imshow(piv, text_auto=".0%", aspect="auto",
                            color_continuous_scale="Purples")
            fig.update_layout(xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, width="stretch")

        st.subheader("Cảm xúc trung bình theo số sao")
        merged = emo.merge(df[["review_id", "rating"]], on="review_id", how="inner")
        emo_cols = [c for c in emo.columns if c.startswith("emo_")]
        show = ["emo_admiration", "emo_joy", "emo_gratitude", "emo_disappointment",
                "emo_anger", "emo_annoyance", "emo_disgust", "emo_sadness"]
        show = [c for c in show if c in emo_cols]
        g = merged.groupby("rating")[show].mean().reset_index()
        gm = g.melt(id_vars="rating", var_name="emotion", value_name="score")
        gm["emotion"] = gm["emotion"].str.replace("emo_", "")
        fig = px.line(gm, x="rating", y="score", color="emotion", markers=True)
        fig.update_layout(xaxis_title="Số sao", yaxis_title="Xác suất TB")
        st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# TAB 5 — Predictive (Sharda Ch.5: data-mining classification)               #
# --------------------------------------------------------------------------- #
with tab_pred:
    st.subheader("Dự báo review tiêu cực (rating ≤ 2)")
    st.caption("Logistic Regression — minh hoạ quy trình khai phá dữ liệu "
               "(business understanding → data prep → model → evaluation).")
    if len(df) < 200 or df["rating"].le(2).sum() < 20:
        st.info("Cần thêm dữ liệu (nới bộ lọc) để huấn luyện mô hình ổn định.")
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (roc_auc_score, confusion_matrix,
                                      classification_report, RocCurveDisplay)

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
        m[1].metric("Tỷ lệ tiêu cực", f"{y.mean()*100:.1f}%")
        m[2].metric("Mẫu test", f"{len(yte):,}")

        a, b = st.columns(2)
        with a:
            st.caption("Ma trận nhầm lẫn")
            cm = confusion_matrix(yte, pred)
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                            x=["Pred: Không", "Pred: Tiêu cực"],
                            y=["Thực: Không", "Thực: Tiêu cực"])
            st.plotly_chart(fig, width="stretch")
        with b:
            st.caption("Trọng số đặc trưng (ảnh hưởng)")
            coef = pd.DataFrame({"feature": feats, "coef": clf.coef_[0]})
            coef = coef.sort_values("coef")
            fig = px.bar(coef, x="coef", y="feature", orientation="h",
                         color="coef", color_continuous_scale="RdBu")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")

        with st.expander("Báo cáo phân loại chi tiết"):
            rep = classification_report(yte, pred, output_dict=True,
                                        target_names=["Không", "Tiêu cực"])
            st.dataframe(pd.DataFrame(rep).T.round(3))


# --------------------------------------------------------------------------- #
# TAB 6 — Prescriptive (Sharda Ch.8: từ phân tích → quyết định hành động)     #
# --------------------------------------------------------------------------- #
with tab_presc:
    st.subheader("Đề xuất hành động — sản phẩm nào cần ưu tiên xử lý?")
    st.caption("Prescriptive analytics: chuyển insight thành quyết định — "
               "chấm điểm rủi ro, ma trận quyết định và tối ưu nguồn lực có ràng buộc.")

    min_np = st.slider("Số review tối thiểu / sản phẩm", 3, 50, 8, key="presc_min")
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
        st.info("Không có sản phẩm nào đạt ngưỡng số review. Giảm ngưỡng ở trên.")
    else:
        prod["neg_share"] = prod["neg_count"] / prod["reviews"]
        prod["name"] = prod["name"].fillna(prod["parent_asin"]).astype(str).str.slice(0, 45)

        # --- Decision rules → action tier (Ch.8 decision analysis) ----------
        vol_med = prod["reviews"].median()

        def action_tier(r):
            if r["avg_rating"] < 3.0 and r["reviews"] >= vol_med:
                return "🔴 Ưu tiên xử lý"
            if r["avg_rating"] < 3.5:
                return "🟡 Cải thiện / theo dõi"
            if r["avg_rating"] >= 4.0 and r["reviews"] >= vol_med:
                return "🟢 Khai thác / quảng bá"
            return "⚪ Duy trì"

        prod["action"] = prod.apply(action_tier, axis=1)
        TIER_COLORS = {"🔴 Ưu tiên xử lý": "#C0392B", "🟡 Cải thiện / theo dõi": "#E1A100",
                       "🟢 Khai thác / quảng bá": "#2E8B57", "⚪ Duy trì": "#9AA0A6"}

        k1, k2, k3 = st.columns(3)
        urgent = prod["action"].eq("🔴 Ưu tiên xử lý").sum()
        k1.metric("Sản phẩm ưu tiên xử lý", f"{urgent}")
        k2.metric("Review tiêu cực (đã lọc)", f"{int(prod['neg_count'].sum()):,}")
        k3.metric("Sản phẩm đánh giá", f"{len(prod):,}")

        st.divider()
        # --- Decision matrix: reach (volume) × satisfaction (rating) --------
        st.subheader("Ma trận quyết định: Độ phủ × Mức hài lòng")
        fig = px.scatter(prod, x="reviews", y="avg_rating", size="neg_count",
                         color="action", color_discrete_map=TIER_COLORS,
                         hover_name="name", size_max=40,
                         labels={"reviews": "Số review (độ phủ)",
                                 "avg_rating": "Rating TB (hài lòng)"})
        fig.add_hline(y=3.5, line_dash="dot", line_color="grey")
        fig.add_vline(x=vol_med, line_dash="dot", line_color="grey")
        fig.update_layout(legend_title="Hành động đề xuất", yaxis_range=[1, 5])
        st.plotly_chart(fig, width="stretch")
        st.caption("Góc phải-dưới (nhiều review, rating thấp) = tác động lớn nhất → "
                   "xử lý trước. Góc phải-trên = sản phẩm nên quảng bá.")

        st.divider()
        # --- Resource allocation optimization (Ch.8: tối ưu có ràng buộc) ---
        st.subheader("Tối ưu nguồn lực: tập trung vào đâu để giảm nhiều than phiền nhất?")
        rank = prod.sort_values("neg_count", ascending=False).reset_index(drop=True)
        total_neg = int(rank["neg_count"].sum())
        max_k = int(len(rank))
        cap = st.slider("Nguồn lực: số sản phẩm có thể xử lý trong kỳ", 1,
                        max_k, min(5, max_k), key="presc_cap")
        chosen = rank.head(cap)
        covered = int(chosen["neg_count"].sum())
        cov_pct = covered / total_neg * 100 if total_neg else 0

        c1, c2 = st.columns([2, 3])
        with c1:
            st.metric("% review tiêu cực được giải quyết",
                      f"{cov_pct:.0f}%",
                      help=f"Xử lý {cap}/{max_k} sản phẩm → giải quyết "
                           f"{covered:,}/{total_neg:,} review tiêu cực.")
            st.caption(f"Chỉ cần tập trung **{cap}/{max_k}** sản phẩm "
                       f"để xử lý **{cov_pct:.0f}%** than phiền — nguyên lý Pareto.")
        with c2:
            rank["cum_pct"] = rank["neg_count"].cumsum() / total_neg * 100 if total_neg else 0
            rank["idx"] = range(1, len(rank) + 1)
            fig = px.area(rank, x="idx", y="cum_pct",
                          labels={"idx": "Số sản phẩm ưu tiên (giảm dần)",
                                  "cum_pct": "% than phiền tích luỹ"})
            fig.add_vline(x=cap, line_dash="dash", line_color="#C0392B")
            fig.update_traces(line_color="#4C78A8")
            fig.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig, width="stretch")

        st.markdown("**Danh sách hành động ưu tiên (kỳ này):**")
        st.dataframe(
            chosen[["name", "category", "reviews", "avg_rating",
                    "neg_count", "neg_share", "action"]]
            .rename(columns={"name": "Sản phẩm", "category": "Danh mục",
                             "reviews": "Reviews", "avg_rating": "Rating TB",
                             "neg_count": "Review tiêu cực", "neg_share": "Tỷ lệ TC",
                             "action": "Hành động"}).round(2),
            hide_index=True, width="stretch")

        # --- Category-level auto recommendation (data storytelling) ---------
        st.divider()
        st.subheader("Khuyến nghị theo danh mục")
        cat_sum = (d.groupby("category")
                     .agg(reviews=("review_id", "count"),
                          avg_rating=("rating", "mean"),
                          neg_share=("is_neg", "mean")).reset_index())
        for _, r in cat_sum.sort_values("neg_share", ascending=False).iterrows():
            if r["neg_share"] >= 0.15:
                msg = (f"⚠️ **{r['category']}** — tỷ lệ tiêu cực cao "
                       f"({r['neg_share']*100:.0f}%). Rà soát chất lượng & mô tả sản phẩm.")
            elif r["avg_rating"] >= 4.3:
                msg = (f"✅ **{r['category']}** — hài lòng cao "
                       f"(rating {r['avg_rating']:.2f}). Đẩy mạnh marketing/review khuyến khích.")
            else:
                msg = (f"➡️ **{r['category']}** — ổn định (rating {r['avg_rating']:.2f}, "
                       f"tiêu cực {r['neg_share']*100:.0f}%). Duy trì & theo dõi.")
            st.markdown(msg)


st.sidebar.divider()
with st.sidebar.expander("ℹ️ Khung kiến thức (Sharda)"):
    st.markdown(
        "- **Ch.3/4** Tiền xử lý, kho dữ liệu, dashboard thông tin\n"
        "- **Ch.5** Phân loại khai phá dữ liệu (tab ⑤)\n"
        "- **Ch.6** Text & sentiment/emotion analytics (tab ③④)\n"
        "- **Ch.8** Prescriptive — đề xuất & tối ưu hành động (tab ⑥)\n"
        "- Dữ liệu: Amazon Reviews 2023 · Cảm xúc: GoEmotions (cầu nối đồ án NLP)")
