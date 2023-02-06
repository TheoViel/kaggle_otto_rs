import numpy as np
import pandas as pd
from params import GT_FILE


def get_coverage(preds, gts):
    """
    Gets the coverage between predictions and ground truth.

    Args:
        preds (list of lists): Predictions for each session.
        gts (list of lists): Ground truth for each session.

    Returns:
        int: Number of predictions,
        int: Number of ground truths,
        int: Number of found ground truths.
    """
    n_preds = 0
    n_gts = 0
    n_found = 0

    for i in range(len(preds)):
        n_preds += len(preds[i])
        if not isinstance(gts[i], (list, np.ndarray)):
            continue

        n_gts += min(20, len(gts[i]))
        n_found += min(20, len(set(list(gts[i])).intersection(set(list(preds[i])))))

    return n_preds, n_gts, n_found


def evaluate(df_val, target, verbose=1):
    """
    Evaluates results.

    Args:
        df_val (cudf DataFrame): Results.
        target (str): Target, in ["gt_orders", "gt_carts", "gt_clicks"]
        verbose (int, optional): Verboisity. Defaults to 1.

    Returns:
        float: Recall@20
    """
    preds = df_val[["session", "candidates", "pred"]].copy()
    preds = preds.sort_values(["session", "pred"], ascending=[True, False])
    preds = (
        preds[["session", "candidates", "pred"]]
        .groupby("session")
        .agg(list)
        .reset_index()
    )
    try:
        preds = preds.to_pandas()
    except Exception:
        pass
    preds["candidates"] = preds["candidates"].apply(lambda x: x[:20])

    gt = pd.read_parquet(GT_FILE)
    preds = preds.merge(
        gt[gt["type"] == target[3:]].drop("type", axis=1), how="left"
    ).rename(columns={"ground_truth": target})

    n_preds, n_gts, n_found = get_coverage(
        preds["candidates"].values, preds[target].values
    )

    if verbose:
        print(f"\n-> {target}  -  Recall : {n_found / n_gts :.4f}\n")
    return n_found / n_gts
