# 📊 Amazon Reviews BI Dashboard

Một dashboard thông minh kinh doanh (BI) phân tích đánh giá sản phẩm Amazon,
đi từ mô tả → chẩn đoán → dự đoán → đề xuất hành động, dựa trên khung lý thuyết
của *Sharda — Business Intelligence, Analytics, and Data Science*. Dự án dùng lại
mô hình phân loại cảm xúc **GoEmotions** làm "động cơ" text analytics.

> 📄 Xem kết quả phân tích chi tiết (số liệu + biểu đồ thật) tại **[RESULTS.md](RESULTS.md)**.

> Ban đầu được xây dựng như đồ án cuối kỳ môn **Business Intelligence & Predictive
> Analytics**, sau đó phát triển thêm thành một dự án BI hoàn chỉnh.

---

## 1. Câu chuyện phân tích (analytics story)

> Hàng nghìn review Amazon là dữ liệu phi cấu trúc. Dashboard biến chúng thành
> thông tin ra quyết định: sản phẩm/danh mục nào đang được yêu thích, sắc thái
> khách hàng thay đổi ra sao theo thời gian, cảm xúc nào đi kèm review tiêu cực,
> và **dự báo** review tiêu cực từ đặc trưng có sẵn.

## 2. Ánh xạ tới sách Sharda

| Chương | Nội dung sách | Hiện thực trong dự án |
|--------|---------------|------------------------|
| **Ch.3** | Nature of data, preprocessing, descriptive statistics | `src/ingest.py` — làm sạch, chuẩn hoá, đặc trưng (độ dài, word count) |
| **Ch.4** | Data warehousing, OLAP, **information dashboards** | Pipeline raw→processed; dashboard 3 tầng (KPI → so sánh → drill-down) |
| **Ch.5** | Predictive analytics — data-mining classification | Tab ⑤: Logistic Regression dự báo review tiêu cực + đánh giá mô hình |
| **Ch.6** | Text, sentiment & **emotion analytics** | VADER sentiment (toàn mẫu) + GoEmotions 28 cảm xúc (tab ③④) |
| **Ch.8** | Prescriptive — optimization & decision analysis | Tab ⑥: chấm rủi ro, ma trận quyết định, tối ưu nguồn lực (Pareto) |

Thiết kế dashboard bám nguyên tắc Ch.4: *thông tin ở 3 mức độ*, *chọn đúng
visual construct*, *guided analytics* (bộ lọc tương tác ở sidebar).

## 3. Dữ liệu

- **Nguồn:** [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) (McAuley-Lab, HuggingFace).
- **Phạm vi:** mẫu **cân bằng ~4.000 review/danh mục** × 4 danh mục
  (Digital Music, Gift Cards, Magazine Subscriptions, Subscription Boxes) ≈ **16.000 review**.
- **Trường chính:** rating, tiêu đề, nội dung, sản phẩm (parent_asin), giá,
  store, helpful votes, verified, thời gian (1998–2023).

## 4. Kiến trúc

```
data/raw/         ← jsonl tải từ HuggingFace (gitignored)
data/processed/   ← reviews.parquet (+ sentiment), reviews_emotions.parquet
src/
  config.py        ← tham số tập trung (danh mục, kích thước mẫu, model)
  ingest.py        ← Stage 1: tải · làm sạch · join meta · VADER sentiment
  score_emotions.py← Stage 2: GoEmotions 28 cảm xúc (mẫu phân tầng)
app.py             ← Dashboard Streamlit (5 tab)
```

## 5. Cách chạy

```bash
pip install -r requirements.txt

python -m src.ingest          # Stage 1 — tạo data/processed/reviews.parquet
python -m src.score_emotions  # Stage 2 — (tuỳ chọn) tab Cảm xúc

streamlit run app.py          # mở dashboard
```

> Mọi tham số (danh mục, kích thước mẫu, model cảm xúc) nằm trong `src/config.py`.

## 6. Sáu tab của dashboard

1. **Tổng quan** — KPI, phân bố rating, cơ cấu sắc thái, khối lượng theo thời gian.
2. **Danh mục & Sản phẩm** — rating theo danh mục, top/bottom sản phẩm, helpful vs rating.
3. **Sắc thái & Văn bản** — sentiment vs số sao, xu hướng theo thời gian, đọc thử review.
4. **Cảm xúc (GoEmotions)** — 28 cảm xúc, cảm xúc × danh mục, cảm xúc theo số sao.
5. **Dự báo** — phân loại review tiêu cực (ROC-AUC, confusion matrix, trọng số đặc trưng).
6. **Đề xuất hành động** — chấm điểm rủi ro sản phẩm, ma trận quyết định (độ phủ × hài lòng), tối ưu nguồn lực theo Pareto + khuyến nghị theo danh mục.

> 🇬🇧 Bản tiếng Anh của dashboard: `streamlit run app_en.py`.

## 7. Hướng mở rộng

- Topic modeling (LDA) cho lý do khen/chê (Ch.6).
- Phân khúc sản phẩm bằng clustering (Ch.5).
- Thay GoEmotions bằng đúng checkpoint BERT fine-tuned từ đồ án NLP.
- Cảnh báo thời gian thực khi tỷ lệ tiêu cực vượt ngưỡng (Ch.4 real-time BI).
