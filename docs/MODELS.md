# Các Model Trong Hệ Thống

Tổng quan bốn model hiện có trong `src/models/`. Tất cả đều kế thừa cùng một
interface `BaseRecommender` nên `src/training/train.py` có thể hoán đổi bằng
flag `--model`.

- [ARCHITECTURE.md](ARCHITECTURE.md) giải thích model được nạp ở đâu trong
  serving path.
- [OPERATIONS.md](OPERATIONS.md) hướng dẫn retrain + reload.

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
`(item_idx, score)`. `exclude_seen=True` ẩn những phim user đã rate — nhờ vậy
API không bao giờ gợi ý lại phim user đã xem.

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

**Khi nào dùng.** Làm sàn (floor) — bất kỳ model nào bạn xây cũng phải đánh
bại Random trên mọi metric, nếu không model đang có lỗi.

**Điểm trên ml-latest-small** (tham khảo): P@10 ≈ 0.002, NDCG@10 ≈ 0.003.

## 2. `PopularityRecommender` — non-personalized baseline

File: `src/models/baseline.py:78`

| Thuộc tính | Giá trị |
|---|---|
| Loại | Baseline (non-personalized) |
| Input | Interaction matrix |
| Train | O(n_items) để sum popularity |
| Recommend | O(n_items) |

**Cách hoạt động.** Xếp hạng item theo tổng số interaction. Mọi user đều
nhận cùng danh sách (trừ các phim đã xem). Score chuẩn hoá `pop(i) / max_pop`.

**Khi nào dùng.**
- Baseline mạnh bất ngờ trên MovieLens — vì phim hit thật sự phổ biến với
  phần lớn user.
- Cold-start fallback: user mới chưa có rating thì serving có thể trả về
  popular items (hiện `src/serving/recommender.py` đã có logic này).

**Điểm trên ml-latest-small** (tham khảo): P@10 ≈ 0.03 — mốc "pass" tối thiểu
cho mọi model CF/content-based.

## 3. `ALSRecommender` — collaborative filtering chính

File: `src/models/collaborative.py:19`

| Thuộc tính | Giá trị |
|---|---|
| Loại | Matrix Factorization (implicit feedback) |
| Thư viện | `implicit.als.AlternatingLeastSquares` |
| Hyperparams | `factors=200`, `regularization=0.01`, `iterations=50`, `alpha=100.0`, `use_bm25=True`, `bm25_k1=100.0`, `bm25_b=0.8` |
| Defaults trong config | `src/config.py:29-33` |

Đây là model production hiện tại. Lưu ý hai kỹ thuật nâng độ chính xác:

### BM25 weighting (mặc định bật)

Trước khi fit, interaction matrix đi qua
`bm25_weight(K1=bm25_k1, B=bm25_b)` từ `implicit.nearest_neighbours`. BM25
down-weight các item cực phổ biến — nếu không, ALS sẽ "collapse" về blockbuster
và quên item đuôi dài.

Trên ml-latest-small, bật BM25 nâng P@10 từ **0.039 → 0.054** (+38%).

### Confidence-weighted loss (Hu–Koren–Volinsky 2008)

Ma trận (đã BM25 hoặc raw) được nhân với `alpha` trước khi fit:
`confidence = weighted * alpha`. ALS coi giá trị lớn = "tín hiệu mạnh hơn" —
chuẩn thực hành cho implicit feedback.

### Quy trình fit

1. Cache `user_seen[uid]` từ CSR indices.
2. `bm25_weight` → `* alpha` → float32 để tiết kiệm RAM.
3. `AlternatingLeastSquares.fit(confidence_matrix)` — mỗi iteration update
   `user_factors` và `item_factors` luân phiên bằng least squares.

### Recommend

`scores = user_factors[uid] @ item_factors.T`. Set `-inf` cho seen items,
lấy `argpartition` top-n rồi sort giảm dần.

### `similar_items(item_idx, n)`

Dùng `implicit.als.similar_items` — cosine trên item factors. Endpoint
`GET /similar/{movie_id}` gọi hàm này.

**Điểm trên ml-latest-small** (hiện tại): P@10 ≈ 0.054, NDCG@10 ≈ 0.06,
Hit@10 ≈ 0.35, MRR ≈ 0.12.

## 4. `ContentBasedRecommender` — cold-start fallback

File: `src/models/content_based.py:18`

| Thuộc tính | Giá trị |
|---|---|
| Loại | Content-based (cosine trên item features) |
| Thư viện | `sklearn.metrics.pairwise.cosine_similarity` |
| Input thêm | `item_features` (genres multi-hot + year + popularity) |

### Cách hoạt động

1. **User profile.** Với mỗi user, lấy weighted average của feature vector
   các phim họ đã xem:

   ```
   profile[u] = Σ (rating[u, i] · features[i]) / Σ rating[u, i]
   ```

2. **Recommend.** Cosine similarity giữa `profile[u]` và từng item feature,
   loại seen items, lấy top-n.

### Khi nào dùng

- **Cold-start items.** Phim mới chưa có interaction vẫn được gợi ý nếu
  genre trùng với sở thích user.
- **Cold-start users tương đối.** User có vài rating là đủ có profile, trong
  khi ALS cần nhiều hơn.
- **Kết quả không bằng ALS trên warm users** — thuần genre không đủ biểu đạt
  sở thích tinh tế (hai phim hành động rất khác nhau vẫn nhìn giống nhau).

### Giới hạn

- Nếu user chưa có interaction nào → profile = 0 → trả về list rỗng. Serving
  phải fallback sang PopularityRecommender.
- Dense cosine trên (n_users, n_features) × (n_items, n_features) — OK với
  ml-latest-small nhưng scale kém trên ml-25m.

## Lựa chọn model trong training

File: `src/training/train.py`

```bash
python -m src.training.train --model random
python -m src.training.train --model popularity
python -m src.training.train --model als          # mặc định, production
python -m src.training.train --model content_based
```

Mỗi run log vào MLflow: tất cả hyperparams + 7 metrics
(precision, recall, ndcg, f1, hit_rate, mrr, coverage). Artifact `model.pkl` +
`model_meta.json` được ghi vào `$RECSYS_ARTIFACTS_DIR`.

## So sánh nhanh trên ml-latest-small

| Model | P@10 | NDCG@10 | Hit@10 | MRR | Coverage | Personalized? |
|---|---|---|---|---|---|---|
| Random | ~0.002 | ~0.003 | ~0.02 | ~0.01 | ~1.0 | Có (ngẫu nhiên) |
| Popularity | ~0.03 | ~0.035 | ~0.25 | ~0.08 | ~0.0015 | Không |
| **ALS + BM25** | **~0.054** | **~0.06** | **~0.35** | **~0.12** | **~0.15** | Có |
| Content-based | ~0.025 | ~0.03 | ~0.18 | ~0.05 | ~0.4 | Có (vừa) |

Con số minh họa — thay đổi tuỳ split/seed. Xem MLflow runs để có số mới nhất.

## Thêm model mới

1. Tạo class kế thừa `BaseRecommender`, implement `fit` + `recommend`.
2. Thêm nhánh trong `src/training/train.py` (hiện đang `if/elif` theo
   `model_type`).
3. Thêm branch serialization nếu artifact khác cấu trúc ALS/content.
4. Cập nhật `src/serving/recommender.py:load_model` để biết cách nạp.
5. Viết unit test ở `tests/unit/test_<model>.py` — luôn check signature
   `fit(interaction_matrix, **kwargs)` trước khi viết test.

## Đọc thêm

- [ARCHITECTURE.md](ARCHITECTURE.md) — model nằm ở đâu trong train → serve flow.
- [OPERATIONS.md](OPERATIONS.md) — retrain + reload runbook.
- Hu, Koren, Volinsky (2008), *Collaborative Filtering for Implicit Feedback
  Datasets* — lý thuyết đằng sau `alpha` weighting.
