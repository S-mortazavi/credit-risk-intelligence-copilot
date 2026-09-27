import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def evaluate_probabilities(y_true, y_prob) -> dict:
    """
    Evaluate probabilistic binary classification predictions.

    Parameters
    ----------
    y_true:
        Binary observed outcomes.

    y_prob:
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Discrimination and probability-quality metrics.
    """

    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "mean_pd": np.mean(y_prob),
        "observed_rate": np.mean(y_true),
    }