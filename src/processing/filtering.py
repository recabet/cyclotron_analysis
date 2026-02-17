"""
Outlier detection and filtering using IQR (Interquartile Range) rule.
"""
import numpy as np
from typing import Tuple


def iqr_outlier_filter(values: np.ndarray, k: float = 1.5) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Identify outliers using the Tukey IQR rule.

    Outliers are values < Q1 - k*IQR or > Q3 + k*IQR.

    Args:
        values: 1D array of values to filter
        k: IQR multiplier (standard = 1.5 for extreme outliers)

    Returns:
        keep_mask: Boolean array, True for non-outliers
        drop_mask: Boolean array, True for outliers
        lower_bound: Lower threshold
        upper_bound: Upper threshold
    """
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    drop_mask = (values < lower) | (values > upper)
    keep_mask = ~drop_mask

    return keep_mask, drop_mask, lower, upper