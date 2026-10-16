"""PA27 feature engineering: only information strictly before the target hour."""
import pandas as pd

from config.experiment import TARGET_COLUMN

WEATHER_COLUMNS = (
    "temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"
)


def make_features(df, representation="A"):
    """Return chronologically aligned X and y with strictly past-only features.

    Call separately for each city, on the full chronological series BEFORE splitting
    warm-up/evaluation. This preserves genuine history at the evaluation boundary.
    Rows with insufficient history or missing values are removed explicitly.
    """
    if representation not in ("A", "B"):
        raise ValueError("representation must be 'A' or 'B'")
    required = ["timestamp_utc", *WEATHER_COLUMNS, TARGET_COLUMN]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not df["timestamp_utc"].is_monotonic_increasing:
        raise ValueError("Data must be ordered chronologically")
    if df["timestamp_utc"].duplicated().any():
        raise ValueError("Duplicate timestamps")
    if "city" in df.columns and df["city"].nunique() != 1:
        raise ValueError("Build features separately for each city")

    result = df[["timestamp_utc", TARGET_COLUMN]].copy()
    for col in WEATHER_COLUMNS:
        result[f"{col}_lag1"] = df[col].shift(1)

    if representation == "B":
        result["temperature_change_lag"] = df["temperature_2m"].shift(1) - df["temperature_2m"].shift(2)
        result["temperature_mean_6h"] = df["temperature_2m"].shift(1).rolling(6, min_periods=6).mean()
        result["temperature_std_6h"] = df["temperature_2m"].shift(1).rolling(6, min_periods=6).std()
        result["cold_frequency_24h"] = df[TARGET_COLUMN].shift(1).rolling(24, min_periods=24).mean()

    feature_cols = [c for c in result.columns if c not in ("timestamp_utc", TARGET_COLUMN)]
    result = result.dropna(subset=feature_cols + [TARGET_COLUMN]).reset_index(drop=True)
    return result


def split_features(feature_frame):
    """Use existing UTC period selection for warm-up and evaluation."""
    from src.preprocessing import get_warmup, get_evaluation_stream
    return get_warmup(feature_frame), get_evaluation_stream(feature_frame)
