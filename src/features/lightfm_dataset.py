"""Xây dựng dữ liệu đầu vào cho LightFM từ MovieLens raw CSV.

Module này làm 4 việc chính:
1. Đọc ``ratings.csv`` + ``movies.csv`` từ ``data/raw/``.
2. Chia 80/20 random split trên ratings (theo yêu cầu người dùng).
3. Dùng ``lightfm.data.Dataset`` để build interaction matrix + item features.
4. Trả về bundle gồm mọi thứ mà training / serving cần để gọi ``model.predict``.

Item features = genres trong ``movies.csv`` (pipe-delimited). Không cần ``tags.csv``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from lightfm.data import Dataset
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


# --- Tên các feature "không có genre" hay gặp trong MovieLens ---------------
_NO_GENRE_TOKENS = {"(no genres listed)", ""}


@dataclass
class LightFMBundle:
    """Gói tất cả artefact cần cho huấn luyện và inference LightFM."""

    dataset: Dataset
    train_interactions: csr_matrix
    test_interactions: csr_matrix
    item_features: csr_matrix
    user_id_map: dict[int, int]  # original userId -> internal user idx
    item_id_map: dict[int, int]  # original movieId -> internal item idx
    reverse_item_id_map: dict[int, int]  # internal item idx -> original movieId
    reverse_user_id_map: dict[int, int]  # internal user idx -> original userId
    movies: pd.DataFrame  # movieId, title, genres (raw) — phục vụ serving
    all_genres: list[str]


def _parse_genres(value: object) -> list[str]:
    """Tách chuỗi genre dạng ``"Action|Comedy"`` thành ``["Action", "Comedy"]``."""
    if value is None:
        return []
    parts = [g.strip() for g in str(value).split("|")]
    return [g for g in parts if g and g not in _NO_GENRE_TOKENS]


def _collect_all_genres(movies: pd.DataFrame) -> list[str]:
    """Lấy toàn bộ genre duy nhất trong catalog, đã sort để kết quả ổn định."""
    seen: set[str] = set()
    for raw in movies["genres"].dropna():
        seen.update(_parse_genres(raw))
    return sorted(seen)


def build_lightfm_dataset(
    ratings_csv: Path,
    movies_csv: Path,
    test_size: float = 0.2,
    seed: int = 42,
) -> LightFMBundle:
    """Xây ``LightFMBundle`` từ CSV thô.

    Parameters
    ----------
    ratings_csv:
        Đường dẫn tới ``ratings.csv`` (cột: userId, movieId, rating, timestamp).
    movies_csv:
        Đường dẫn tới ``movies.csv`` (cột: movieId, title, genres).
    test_size:
        Tỷ lệ split test (mặc định 0.2 => 80/20).
    seed:
        Random seed cho ``train_test_split`` — đảm bảo reproducible.
    """
    log.info("Đang đọc ratings từ %s", ratings_csv)
    ratings = pd.read_csv(ratings_csv)

    log.info("Đang đọc movies từ %s", movies_csv)
    movies = pd.read_csv(movies_csv)

    # --- Chia train/test 80/20 random ----------------------------------------
    train_df, test_df = train_test_split(
        ratings,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    log.info(
        "Split xong: train=%d, test=%d (test_size=%.2f)", len(train_df), len(test_df), test_size
    )

    # --- Thu thập vũ trụ user / item / genre ---------------------------------
    all_users = ratings["userId"].unique().tolist()
    all_items = movies["movieId"].unique().tolist()
    all_genres = _collect_all_genres(movies)
    log.info(
        "Vũ trụ: users=%d, items=%d, genres=%d", len(all_users), len(all_items), len(all_genres)
    )

    # --- Fit LightFM Dataset -------------------------------------------------
    # Dataset.fit chịu trách nhiệm đánh internal index cho mọi user/item/feature.
    dataset = Dataset()
    dataset.fit(users=all_users, items=all_items, item_features=all_genres)

    # --- Build interaction matrices ------------------------------------------
    # Chỉ dùng cặp (userId, movieId); giữ rating làm weight (LightFM chấp nhận).
    train_interactions, _ = dataset.build_interactions(
        (int(r.userId), int(r.movieId)) for r in train_df.itertuples(index=False)
    )
    test_interactions, _ = dataset.build_interactions(
        (int(r.userId), int(r.movieId)) for r in test_df.itertuples(index=False)
    )

    # --- Build item features (genres) ----------------------------------------
    # Mỗi movie gắn với list genre của chính nó (multi-hot).
    item_features = dataset.build_item_features(
        ((int(row.movieId), _parse_genres(row.genres)) for row in movies.itertuples(index=False)),
        normalize=True,
    )

    # --- Lấy các mapping để serving không phụ thuộc Dataset ------------------
    # Dataset.mapping() trả về: (user_id_map, user_feat_map, item_id_map, item_feat_map)
    user_id_map, _, item_id_map, _ = dataset.mapping()
    # Ép khóa về int để joblib pickle gọn và để API nhận user_id kiểu int thoải mái.
    user_id_map_int = {int(k): int(v) for k, v in user_id_map.items()}
    item_id_map_int = {int(k): int(v) for k, v in item_id_map.items()}
    reverse_item_id_map = {v: k for k, v in item_id_map_int.items()}
    reverse_user_id_map = {v: k for k, v in user_id_map_int.items()}

    log.info(
        "Matrices built: train=%s, test=%s, item_features=%s",
        train_interactions.shape,
        test_interactions.shape,
        item_features.shape,
    )

    return LightFMBundle(
        dataset=dataset,
        train_interactions=train_interactions.tocsr(),
        test_interactions=test_interactions.tocsr(),
        item_features=item_features.tocsr(),
        user_id_map=user_id_map_int,
        item_id_map=item_id_map_int,
        reverse_item_id_map=reverse_item_id_map,
        reverse_user_id_map=reverse_user_id_map,
        movies=movies,
        all_genres=all_genres,
    )
