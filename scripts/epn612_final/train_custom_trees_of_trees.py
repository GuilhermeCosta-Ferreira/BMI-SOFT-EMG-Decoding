"""Feature-type importance aggregation reused by the final experiment."""

import numpy as np


def _feature_type_importance(metadata, values, expected_total):
    """Aggregate using aligned metadata, including zero-importance feature types."""
    values = np.asarray(values, dtype=float)
    if (
        values.shape != (len(metadata),)
        or not np.isfinite(values).all()
        or (values < 0).any()
    ):
        raise ValueError(
            "Feature importance must be finite, nonnegative, and aligned with inputs"
        )
    raw = metadata.assign(importance=values)
    ranking = (
        raw.groupby("feature_type", as_index=False, sort=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    if (
        ranking.feature_type.duplicated().any()
        or set(ranking.feature_type) != set(metadata.feature_type)
        or len(ranking) != metadata.feature_type.nunique()
        or not np.isclose(values.sum(), expected_total, atol=1e-10, rtol=1e-8)
        or not np.isclose(
            ranking.importance.sum(), expected_total, atol=1e-10, rtol=1e-8
        )
    ):
        raise ValueError(
            "Feature-type aggregation lost inputs or has an unexpected importance sum"
        )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking
