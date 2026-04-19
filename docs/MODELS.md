# Các Model Trong Hệ Thống

Tổng quan các model trong `src/models/`. Tất cả đều kế thừa interface
`BaseRecommender` nên `src/training/train.py` có thể hoán đổi bằng flag
`--model`.

- [LIGHTFM_ALGORITHM.md](LIGHTFM_ALGORITHM.md) — giải thích chi tiết thuật toán,
  WARP loss, item features, inference, và hyperparameter tuning.
- [ARCHITECTURE.md](ARCHITECTURE.md) — model được nạp ở đâu trong serving path.
- [OPERATIONS.md](OPERATIONS.md) — retrain + reload runbook.

## Interface chung — `BaseRecommender`

File: `src/models/baseline.py:16`

```python
class BaseRecommender(ABC):
    def fit(self, interaction_matrix: csr_matrix, **kwargs) -> None: ...
    def recommend(
        self, user_idx: int, n: int = 10, exclude_seen: bool = True,
    ) -> list[tuple[int, float]]: ...
```

Mọi model nhận một ma trận `users × items` dạng CSR và trả về top-*n* cặp
`(item_idx, score)`. `exclude_seen=True` ẩn những phim user đã rate.

---

## 1. `RandomRecommender` — lower-bound baseline

File: `src/models/baseline.py:33`

| Thuộc tính | Giá trị |
|---|---|
| Loại | Baseline (non-personalized) |
| Input | Chỉ cần interaction matrix |
| Độ phức tạp train | O(n_users × avg_items) để cache "seen" |
| Độ phức tạp recommend | O(n_items) |

**Cách hoạt động.** Mỗi user, lấy ngẫu nhiên *n* item chưa tương tác. Score
uniform `1/n`. Dùng `np.random.default_rng(seed=42)` để kết quả lặp lại được.

**Khi nào dùng.** Làm sàn (floor) — bất kỳ model nào cũng phải đánh bại Random
trên mọi metric, nếu không model đang có lỗi.

---

## 2. `PopularityRecommender` — non-personalized baseline

File: `src/models/baseline.py:78`

| Thuộc tính | Giá trị |
|---|---|
| Loại | Baseline (non-personalized) |
| Input | Interaction matrix |
| Train | O(n_items) để sum popularity |
| Recommend | O(n_items) |

**Cách hoạt động.** Xếp hạng item theo tổng số interaction. Mọi user đều
nhận cùng danh sách (trừ các phim đã xem). Score chuẩn hóa `pop(i) / max_pop`.

**Khi nào dùng.**
- Baseline mạnh bất ngờ trên MovieLens — vì phim hit thật sự phổ biến.
- **Cold-start fallback**: `src/serving/recommender.py` tự động trả về popular
  items khi user mới chưa có rating.

---

## 3. `LightFMRecommender` — model production chính

File: `src/models/lightfm_hybrid.py`

| Thuộc tính | Giá trị |
|---|---|
| Loại | Hybrid Matrix Factorization (CF + Content) |
| Thư viện | `lightfm>=1.17` |
| Loss | WARP (Weighted Approximate-Rank Pairwise) |
| Hyperparams | `no_components=64`, `lr=0.05`, `epochs=30`, `num_threads=4` |
| Item features | Genres từ `movies.csv` (pipe-delimited, multi-hot) |
| Split | 80/20 random (`test_size=0.2`, `seed=42`) |

### Tại sao LightFM thay thế ALS

| | ALS (đã xóa) | LightFM (hiện tại) |
|---|---|---|
| Interaction matrix | ✅ | ✅ |
| Item features (genres) | ❌ | ✅ |
| Cold-start item | ❌ | Tốt hơn (genre embedding) |
| Loss | Weighted square | WARP — tối ưu Precision@K |
| Đánh giá chuẩn | Tự viết | `lightfm.evaluation` API |

### Cách hoạt động nhanh

Mỗi item được biểu diễn qua tổng embedding của genres:

```
p(Inception) = e(Action) + e(Sci-Fi) + e(Thriller)
```

Score `ŷ(u, i) = q_u · p_i + b_u + b_i`, tối ưu bằng WARP để
Precision@10 cao nhất.

> Xem [LIGHTFM_ALGORITHM.md](LIGHTFM_ALGORITHM.md) để hiểu chi tiết.

### Recommend flow

```python
scores = model.predict(
    user_ids=user_idx,
    item_ids=np.arange(n_items),
    item_features=item_features,   # phải truyền — dữ liệu genres
    num_threads=4,
)
# argpartition top-K → sort → map về movieId
```

### Similar Items

Cosine similarity trên item embedding matrix được cache lúc load:

```python
item_biases, item_embeddings = model.get_item_representations(features=item_features)
sims = (item_embeddings @ target) / (norms * target_norm)
```

### Kết quả đánh giá (ml-latest-small, tham khảo)

| Metric | Giá trị |
|---|---|
| Precision@10 | ~0.08 |
| Recall@10 | ~0.04 |
| AUC | ~0.92 |
| F1 | ~0.05 |

Số thực tế thay đổi theo split/seed — xem MLflow UI tại `http://localhost:5000`.

---

## Lựa chọn model khi train

File: `src/training/train.py`

```bash
python -m src.training.train --model lightfm    # production (mặc định)
python -m src.training.train --model popular    # baseline
python -m src.training.train --model random     # lower-bound
```

Mỗi run log vào MLflow: tất cả hyperparams + metrics (precision, recall, auc,
f1). Artifact `model.pkl` + `model_meta.json` được ghi vào `$RECSYS_ARTIFACTS_DIR`.

## So sánh nhanh trên ml-latest-small

| Model | P@10 | AUC | Personalized? | Ghi chú |
|---|---|---|---|---|
| Random | ~0.002 | — | Có (ngẫu nhiên) | Lower bound |
| Popularity | ~0.03 | — | Không | Cold-start fallback |
| **LightFM WARP** | **~0.08** | **~0.92** | **Có** | **Production** |

## Thêm model mới

1. Tạo class kế thừa `BaseRecommender`, implement `fit` + `recommend`.
2. Thêm nhánh `elif model_type == "..."` trong `src/training/train.py`.
3. Thêm serialization nếu artifact cần cấu trúc khác trong `_build_artifact`.
4. Cập nhật `src/serving/recommender.py:load` để load artifact mới.
5. Viết unit test tại `tests/unit/test_<model>.py` — luôn verify signature
   `fit(interaction_matrix, **kwargs)` trước.

## Đọc thêm

- [LIGHTFM_ALGORITHM.md](LIGHTFM_ALGORITHM.md) — lý thuyết WARP, item features,
  cold-start, hyperparameter tuning.
- [ARCHITECTURE.md](ARCHITECTURE.md) — train → serve flow.
- [OPERATIONS.md](OPERATIONS.md) — retrain + reload runbook.
- Kula (2015), *Metadata Embeddings for User and Item Cold-start Recommendations*
  — paper gốc LightFM.
