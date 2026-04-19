# Thuật Toán LightFM — Giải Thích Chi Tiết

> Tài liệu này mô tả lý thuyết đằng sau LightFM, cách model được cấu hình trong
> dự án, và cách đọc/hiểu kết quả đánh giá.

## Mục lục

1. [Bối cảnh: từ CF thuần đến Hybrid](#1-bối-cảnh-từ-cf-thuần-đến-hybrid)
2. [Kiến trúc LightFM](#2-kiến-trúc-lightfm)
3. [Hàm mất mát WARP](#3-hàm-mất-mát-warp)
4. [Item Features — vai trò của genres](#4-item-features--vai-trò-của-genres)
5. [Quá trình dự đoán (Inference)](#5-quá-trình-dự-đoán-inference)
6. [Similar Items qua Cosine Similarity](#6-similar-items-qua-cosine-similarity)
7. [Hyperparameters trong dự án](#7-hyperparameters-trong-dự-án)
8. [Metrics đánh giá](#8-metrics-đánh-giá)
9. [Cold-start và Fallback](#9-cold-start-và-fallback)
10. [Luồng code trong dự án](#10-luồng-code-trong-dự-án)
11. [Đọc thêm](#11-đọc-thêm)

---

## 1. Bối cảnh: từ CF thuần đến Hybrid

**Collaborative Filtering (CF) thuần** chỉ dùng ma trận tương tác
`users × items` (ai xem/đánh giá phim gì). Nó học được "user A giống user B"
nhưng hoàn toàn mù quáng với nội dung — nếu một phim mới chưa có ai xem,
CF không thể gợi ý.

**Content-based** ngược lại: dùng đặc trưng của item (genres, diễn viên…) nhưng
bỏ qua tín hiệu xã hội từ hành vi tập thể.

**LightFM (Kula, 2015)** kết hợp cả hai bằng cách biểu diễn mỗi user/item qua
*tổ hợp embedding của features* thay vì một embedding đơn lẻ:

```
embedding(item) = Σ embedding(feature_f)  ∀ feature_f ∈ item
```

Khi một phim mới xuất hiện (chỉ có genres, chưa có interaction), LightFM vẫn có
thể tính điểm tương đồng dựa trên embedding genres đã học từ các phim khác.

---

## 2. Kiến trúc LightFM

### 2.1 Biểu diễn User và Item

Với mỗi user `u` và item `i`, LightFM tính **affinity score** (mức độ phù hợp):

```
ŷ(u, i) = f( q_u · p_i + b_u + b_i )
```

Trong đó:

| Ký hiệu | Ý nghĩa | Kích thước |
|---|---|---|
| `q_u` | Embedding của user `u` | `no_components` |
| `p_i` | Embedding của item `i` | `no_components` |
| `b_u` | Bias của user `u` | scalar |
| `b_i` | Bias của item `i` | scalar |
| `f` | Hàm kích hoạt (sigmoid trong WARP) | — |

### 2.2 Tổng hợp Features thành Embeddings

Đây là phần làm LightFM khác biệt. Thay vì học trực tiếp `q_u` và `p_i`, model
học **embedding cho từng feature** và tổng hợp lại:

```
q_u = Σ  e_f    ∀ f ∈ user_features(u)
      f

p_i = Σ  e_g    ∀ g ∈ item_features(i)
      g
```

Trong dự án này:
- **User features**: không dùng (chỉ có identity feature của mỗi user).
- **Item features**: genres từ `movies.csv`, build dưới dạng multi-hot
  (một phim có thể thuộc nhiều genres).

Ví dụ: phim *Inception* có genres `["Action", "Sci-Fi", "Thriller"]`, nên:

```
p(Inception) = e(Action) + e(Sci-Fi) + e(Thriller)
```

Khi model học được rằng "Sci-Fi" và "Action" hay đi kèm nhau trong sở thích user,
các phim mới chưa từng được xem (nhưng có cùng genre) vẫn có điểm tốt.

### 2.3 So sánh với ALS thuần

| | ALS (trước) | LightFM (hiện tại) |
|---|---|---|
| Interaction matrix | ✅ | ✅ |
| Item features (genres) | ❌ | ✅ |
| Xử lý cold-start item | ❌ | Tốt hơn (dùng genre embedding) |
| Loss mặc định | Weighted square loss | WARP (ranking-aware) |
| Scale | O(n_factors²) per iter | O(n_components × n_features) |

---

## 3. Hàm Mất Mát WARP

**WARP** = *Weighted Approximate-Rank Pairwise loss* (Weston et al., 2011).

### 3.1 Ý tưởng

Với mỗi cặp (user `u`, positive item `i`), WARP ngẫu nhiên lấy mẫu item tiêu cực
`j` cho đến khi tìm được item vi phạm điều kiện:

```
ŷ(u, j) > ŷ(u, i) - 1   (item xấu được score cao hơn item tốt)
```

Số lần lấy mẫu trước khi vi phạm ước lượng **rank** của item tốt trong danh sách.
Gradient update tỉ lệ nghịch với rank đó — phim đang ở vị trí top sẽ bị phạt
nặng hơn nếu xuống hạng.

### 3.2 Tại sao WARP tốt hơn BPR / AUC loss cho task này?

| Loss | Tối ưu cho | Phù hợp |
|---|---|---|
| AUC | Xếp hạng tổng thể | Khi cần phân biệt tốt/xấu toàn bộ list |
| BPR | Pairwise ranking | Phổ biến, nhưng không ưu tiên top-K |
| **WARP** | **Precision@K** | **Gợi ý phim — quan tâm 10 phim đầu nhất** |

Vì endpoint `/recommend` trả về top-10, tối ưu **Precision@K** quan trọng hơn
AUC toàn dải → WARP là lựa chọn phù hợp.

### 3.3 Công thức

Gradient update cho WARP:

```
L(u, i, j) = w(rank(u, i)) · max(0, 1 - ŷ(u,i) + ŷ(u,j))

w(rank) ≈ log(rank + 1)  (xấp xỉ)
```

Update tham số theo SGD với learning rate `lr`:

```
θ ← θ - lr · ∇L
```

---

## 4. Item Features — vai trò của Genres

### 4.1 Build trong code

File: `src/features/lightfm_dataset.py`

```python
# Mỗi phim → list genres (pipe-delimited trong movies.csv)
dataset.fit(
    users=all_users,
    items=all_items,
    item_features=all_genres      # ["Action", "Adventure", ...]
)

item_features = dataset.build_item_features(
    ((movie_id, genres_of_movie) for movie in movies),
    normalize=True                # L1 normalize mỗi hàng
)
```

Kết quả: ma trận `(n_items, n_items + n_genres)`.
- Nửa đầu: **identity features** — mỗi phim có 1 cột riêng (CF component).
- Nửa sau: **genre features** — multi-hot encoding (content component).

`normalize=True` chia đều trọng số khi phim có nhiều genres, tránh phim nhiều
genres bị over-represented.

### 4.2 Genres trong MovieLens

Dataset `ml-latest-small` có 20 genres độc lập:

```
Action, Adventure, Animation, Children, Comedy, Crime,
Documentary, Drama, Fantasy, Film-Noir, Horror, IMAX,
Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western,
(no genres listed)  ← bị loại trong code
```

### 4.3 Tác động thực tế

Với `no_components=64`, model học 64 chiều embedding cho mỗi genre. Các genre
hay xuất hiện cùng nhau (Action + Sci-Fi) sẽ có embedding gần nhau trong không
gian vector, giúp gợi ý chéo giữa các phim cùng "mùi" dù user chưa xem.

---

## 5. Quá Trình Dự Đoán (Inference)

File: `src/serving/recommender.py:recommend`

```python
scores = model.predict(
    user_ids=user_idx,           # internal index của user
    item_ids=np.arange(n_items), # toàn bộ catalog
    item_features=item_features, # bắt buộc — cùng matrix lúc train
    num_threads=4,
)
```

**Output**: array `float32` kích thước `(n_items,)`, mỗi phần tử là affinity
score `ŷ(u, i)` tương ứng.

Các bước xử lý tiếp:
1. Set score `-inf` cho các phim user đã xem (nếu `exclude_seen=True`).
2. `np.argpartition(-scores, top_k)` — O(n) để lấy top_k index.
3. Sort top_k theo score giảm dần.
4. Map internal index → original movieId → metadata.

**Lưu ý**: `item_features` **phải** truyền vào cả lúc train lẫn lúc predict —
nếu thiếu, model chỉ dùng identity feature (mất đi phần content-based). File
`model.pkl` luôn đi kèm `item_features` đã được serialize.

---

## 6. Similar Items qua Cosine Similarity

File: `src/serving/recommender.py:similar_items`

LightFM không có API `similar_items` sẵn như ALS, nhưng ta có thể tự tính qua
**item embeddings** được cache lúc load model:

```python
# Lấy embedding matrix: shape (n_items, no_components)
item_biases, item_embeddings = model.get_item_representations(features=item_features)
```

Với item query `i`, tính cosine similarity với toàn bộ catalog:

```
sim(i, j) = (e_i · e_j) / (||e_i|| · ||e_j||)
```

Vectorized với numpy:

```python
sims = (item_embeddings @ target) / (norms * target_norm)
```

Top-k items có `sim` cao nhất = "phim tương tự về embedding".

**Khác với ALS**: ALS dùng `implicit.als.similar_items` có FAISS/NMSLIB backend
cho approximate nearest neighbor. LightFM phải brute-force O(n_items) — đủ nhanh
với `n_items ≈ 10,000` nhưng cần cache hoặc FAISS nếu scale lên ml-25m.

---

## 7. Hyperparameters Trong Dự Án

File: `src/config.py` + `src/models/lightfm_hybrid.py`

| Tham số | Giá trị | Lý do chọn |
|---|---|---|
| `no_components` | 64 | Balance giữa expressiveness và overfitting trên ml-latest-small (~100k ratings). Thường tune trong [32, 128]. |
| `loss` | `"warp"` | Tối ưu Precision@K — phù hợp với bài toán top-10. |
| `learning_rate` | 0.05 | LightFM SGD mặc định; thường không cần tune nhiều khi đã có WARP. |
| `epochs` | 30 | Đủ hội tụ trên ml-latest-small. Với ml-25m cần tăng lên 50-100. |
| `num_threads` | 4 | Parallelism cho training + inference (OpenMP, chỉ có trên Linux). |
| `test_size` | 0.2 | 80/20 random split theo yêu cầu. |
| `split_seed` | 42 | Reproducibility. |

### Tuning Guide

```
no_components  ↑  → mô hình phức tạp hơn, cần nhiều epoch hơn, dễ overfit
learning_rate  ↑  → hội tụ nhanh nhưng dễ oscillate
epochs         ↑  → đến điểm nào đó metrics trên test ngừng tăng (early stopping)
```

Để so sánh runs, dùng MLflow UI tại `http://localhost:5000`.

---

## 8. Metrics Đánh Giá

File: `src/training/train.py:_evaluate_lightfm`

Dùng API `lightfm.evaluation` — chú ý truyền `train_interactions` để tránh
data leakage (không evaluate trên cặp đã train):

### Precision@K

```
P@K = (số phim relevant trong top-K) / K
```

*Relevant* = user có interaction trong test set. Giá trị `[0, 1]`. Metric chính
vì API trả top-10.

### Recall@K

```
R@K = (số phim relevant trong top-K) / (tổng số phim relevant của user)
```

Cho biết model "bao phủ" bao nhiêu % phim user thích.

### AUC Score

```
AUC = P(score(positive) > score(negative))
```

Xác suất model xếp phim relevant cao hơn phim không relevant. Metric toàn dải,
không phụ thuộc K. LightFM dùng SGD approximation.

### F1 (tính thêm trong code)

```
F1 = 2 · P@K · R@K / (P@K + R@K)
```

Harmonic mean — log vào MLflow để theo dõi trade-off P/R.

### Đọc kết quả

```
============================================================
KẾT QUẢ ĐÁNH GIÁ LIGHTFM HYBRID (k=10)
============================================================
  Precision@10: 0.0823
  Recall@10   : 0.0412
  AUC score   : 0.9241
============================================================
```

- **P@10 > 0.05**: tốt hơn PopularityRecommender → model đang cá nhân hóa.
- **AUC > 0.9**: model phân biệt tốt relevant vs. non-relevant.
- **AUC cao, P@10 thấp**: model học rank tổng thể tốt nhưng top-K kém →
  thử tăng `no_components` hoặc `epochs`.

---

## 9. Cold-start và Fallback

### Cold-start User (user mới, chưa có rating)

Endpoint `/recommend/{user_id}` gặp user_id không có trong training:

```python
# src/serving/recommender.py:recommend
user_idx = self._user_to_idx.get(user_id)
if user_idx is None:
    return self._popular_items[:top_k]  # fallback popular
```

Popular items được tính 1 lần lúc load model từ `user_seen` (item xuất hiện
nhiều lần nhất trong training set). Score chuẩn hóa `count / max_count`.

### Cold-start Item (phim mới, chưa có interaction)

Phim mới chưa có trong `item_id_map` → endpoint `/similar/{movie_id}` raise
`UnknownMovieError`. Phim mới **có thể được gợi ý** nếu thêm vào `item_id_map`
và có genre features — nhưng cần retrain để embedding được cập nhật.

### So sánh với ALS

| | ALS | LightFM |
|---|---|---|
| Cold-start user | Popular fallback | Popular fallback |
| Cold-start item | Không hỗ trợ | Hỗ trợ một phần (genre embedding) |

---

## 10. Luồng Code Trong Dự Án

```
ratings.csv + movies.csv
        │
        ▼
src/features/lightfm_dataset.py
  build_lightfm_dataset()
  └── LightFMBundle (train_interactions, test_interactions,
                     item_features, mappings, movies)
        │
        ▼
src/models/lightfm_hybrid.py
  LightFMRecommender.fit(train_interactions, item_features=...)
  └── LightFM(loss="warp", no_components=64).fit(...)
  └── cache item_embeddings = get_item_representations(item_features)
        │
        ▼
src/training/train.py
  _evaluate_lightfm(model, bundle, k=10)
  _build_artifact(...)  → model.pkl + model_meta.json
  mlflow.log_metrics(...)
        │
        ▼
src/serving/recommender.py
  RecommenderService.load(model_path)
  recommend(user_id)    → model.predict(user_ids, item_ids, item_features)
  similar_items(movie_id) → cosine(item_embeddings)
  get_popular_items()   → fallback cold-start
        │
        ▼
src/serving/routers/{recommend,similar,health}.py
  GET /recommend/{user_id}
  GET /similar/{movie_id}
  GET /health
```

---

## 11. Đọc Thêm

| Tài liệu | Nội dung |
|---|---|
| [Kula (2015)](https://arxiv.org/abs/1507.08439) | Paper gốc LightFM |
| [Weston et al. (2011)](https://arxiv.org/abs/1109.3907) | WARP loss |
| [lightfm.readthedocs.io](https://making.lyst.com/lightfm/docs/home.html) | API docs + tutorials |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Model nằm ở đâu trong train → serve flow |
| [MODELS.md](MODELS.md) | Tổng quan model + metrics benchmark |
| [OPERATIONS.md](OPERATIONS.md) | Retrain + reload runbook |
