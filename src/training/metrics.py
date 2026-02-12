import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics for spectra.
    Args:
        y_true: (N, L) or (L,) ground truth
        y_pred: (N, L) or (L,) predictions
    Returns:
        dict with MSE, MAE, R2, mean Pearson correlation
    """
    Yt = np.atleast_2d(y_true).astype(np.float64)
    Yp = np.atleast_2d(y_pred).astype(np.float64)

    flat_t = Yt.ravel()
    flat_p = Yp.ravel()
    mse = mean_squared_error(flat_t, flat_p)
    mae = mean_absolute_error(flat_t, flat_p)
    r2 = r2_score(flat_t, flat_p)

    # Per-sample Pearson correlation
    A = Yt - Yt.mean(axis=1, keepdims=True)
    B = Yp - Yp.mean(axis=1, keepdims=True)
    num = np.sum(A * B, axis=1)
    den = np.sqrt(np.sum(A * A, axis=1) * np.sum(B * B, axis=1))
    cors = np.divide(num, den, out=np.zeros_like(num), where=(den > 0))

    return {
        "MSE": float(f"{mse:.7f}"),
        "MAE": float(f"{mae:.7f}"),
        "R2": float(f"{r2:.6f}"),
        "Pearson(mean)": float(f"{np.mean(cors):.6f}")
    }