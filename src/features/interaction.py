"""Interaction matrix construction for collaborative filtering models."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse
from numpy.typing import NDArray

log = logging.getLogger(__name__)


def build_interaction_matrix(
    ratings: pd.DataFrame,
    n_users: int,
    n_items: int,
    implicit_threshold: float = 3.5,
) -> tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
    """Build explicit and implicit interaction matrices from ratings.

    Parameters
    ----------
    ratings:
        Must contain ``user_idx``, ``movie_idx``, and ``rating`` columns.
        Both ``user_idx`` and ``movie_idx`` should be zero-based integer
        indices.
    n_users:
        Total number of users (determines matrix row count).
    n_items:
        Total number of items (determines matrix column count).
    implicit_threshold:
        Ratings at or above this value are treated as positive implicit
        feedback (1); everything below becomes 0.

    Returns
    -------
    tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]
        ``(explicit_matrix, implicit_matrix)`` — both of shape
        ``(n_users, n_items)`` in CSR format.  The explicit matrix stores
        actual rating values; the implicit matrix stores binary indicators.
    """
    required_cols = {"user_idx", "movie_idx", "rating"}
    missing = required_cols - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings DataFrame is missing columns: {missing}")

    if len(ratings) == 0:
        log.warning("Empty ratings DataFrame — returning zero matrices")
        zero = scipy.sparse.csr_matrix((n_users, n_items), dtype=np.float32)
        return zero, zero

    row: NDArray[Any] = ratings["user_idx"].to_numpy(dtype=np.int32, copy=False)
    col: NDArray[Any] = ratings["movie_idx"].to_numpy(dtype=np.int32, copy=False)
    rating_vals: NDArray[Any] = ratings["rating"].to_numpy(dtype=np.float32, copy=False)

    # Validate index bounds
    if row.max() >= n_users:
        raise ValueError(
            f"user_idx max ({row.max()}) >= n_users ({n_users}). "
            "Ensure indices are zero-based and n_users is correct."
        )
    if col.max() >= n_items:
        raise ValueError(
            f"movie_idx max ({col.max()}) >= n_items ({n_items}). "
            "Ensure indices are zero-based and n_items is correct."
        )

    explicit_matrix = scipy.sparse.csr_matrix(
        (rating_vals, (row, col)),
        shape=(n_users, n_items),
    )

    # Filter out below-threshold entries before construction so csr_matrix
    # does not store explicit zeros, which would incorrectly mark low-rated
    # items as "seen" in get_user_seen_items().
    implicit_vals = (rating_vals >= implicit_threshold).astype(np.float32)
    pos_mask = implicit_vals > 0
    implicit_matrix = scipy.sparse.csr_matrix(
        (implicit_vals[pos_mask], (row[pos_mask], col[pos_mask])),
        shape=(n_users, n_items),
    )

    log.info(
        "Interaction matrices built: shape=(%d, %d), explicit_nnz=%d, implicit_positive=%d",
        n_users,
        n_items,
        explicit_matrix.nnz,
        int(implicit_matrix.sum()),
    )
    return explicit_matrix, implicit_matrix


def get_user_seen_items(
    interaction_matrix: scipy.sparse.csr_matrix,
) -> dict[int, set[int]]:
    """Map each user to the set of item indices they have interacted with.

    Parameters
    ----------
    interaction_matrix:
        A sparse CSR matrix of shape ``(n_users, n_items)`` where non-zero
        entries indicate an interaction (explicit or implicit).

    Returns
    -------
    dict[int, set[int]]
        Mapping from user index to the set of item indices with non-zero
        entries in *interaction_matrix*.
    """
    if not scipy.sparse.issparse(interaction_matrix):
        raise TypeError("interaction_matrix must be a scipy sparse matrix")

    csr = interaction_matrix.tocsr()
    indptr = csr.indptr
    indices = csr.indices
    seen: dict[int, set[int]] = {}
    for user_idx in range(csr.shape[0]):
        start, end = int(indptr[user_idx]), int(indptr[user_idx + 1])
        if end > start:
            seen[user_idx] = {int(x) for x in indices[start:end]}

    log.info(
        "Built seen-items dict for %d users (out of %d total)",
        len(seen),
        csr.shape[0],
    )
    return seen
