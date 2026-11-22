from datetime import datetime, timezone

import pandas as pd

from ml.preprocessing.features import FEATURE_NAMES, add_time_features, build_feature_frame


def test_training_and_inference_share_calendar_feature_engineering():
    when = datetime(2026, 11, 15, 10, 30, tzinfo=timezone.utc)
    trained_row = add_time_features(pd.DataFrame({"timestamp": [pd.Timestamp("2026-11-15 10:30:00")]}))
    inferred = build_feature_frame(600, 25, 43, when, "UTC")
    assert list(inferred.columns) == FEATURE_NAMES
    for feature in ("hour_sin", "hour_cos", "day_sin", "day_cos"):
        assert inferred.iloc[0][feature] == trained_row.iloc[0][feature]
    assert inferred.iloc[0]["irradiation_ratio"] == 0.6
