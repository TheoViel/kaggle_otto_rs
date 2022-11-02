import numpy as np
from sklearn.metrics import *


def compute_metric(preds, truths):
    """
    Micro f1 on binary arrays.

    Args:
        preds (list of lists of ints): Predictions.
        truths (list of lists of ints): Ground truths.

    Returns:
        float: f1 score.
    """
    # Micro : aggregating over all instances
    preds = np.array(preds)
    truths = np.array(truths)

    return np.sqrt(((truths - preds) ** 2).mean())


def evaluate(preds, truths, eps=1e-4):
    preds = np.array(preds)
    truths = np.array(truths)

    return np.sqrt(((truths - preds) ** 2).mean())