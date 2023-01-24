import numpy as np
import pandas as pd
from params import GT_FILE


def get_hits(prediction, labels, k=20):
    if not isinstance(prediction["clicks"], list):
        prediction["clicks"] = [prediction["clicks"]]

    if labels["clicks"] is not None:
        clicks_hit = float(labels["clicks"] in prediction["clicks"][:k])
    else:
        clicks_hit = 0

    if len(labels["carts"]):
        cart_hits = len(set(prediction["carts"][:k]).intersection(labels["carts"]))
    else:
        cart_hits = 0

    if len(labels["orders"]):
        order_hits = len(set(prediction["orders"][:k]).intersection(labels["orders"]))
    else:
        order_hits = 0

    return {"clicks": clicks_hit, "carts": cart_hits, "orders": order_hits}


def recall(predictions, labels, k=20):
    weights = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}

    total_hits = {"clicks": 0, "carts": 0, "orders": 0}
    total_events = {"clicks": 0, "carts": 0, "orders": 0}

    for pred, label in zip(predictions, labels):
        hits = get_hits(pred, label, k=k)

        total_hits["clicks"] += hits["clicks"]
        total_hits["carts"] += hits["carts"]
        total_hits["orders"] += hits["orders"]

        total_events["clicks"] += label["clicks"] is not None
        total_events["carts"] += len(set(label["carts"]))
        total_events["orders"] += len(set(label["orders"]))

    recalls = {}
    recalls["clicks"] = total_hits["clicks"] / total_events["clicks"]
    recalls["carts"] = total_hits["carts"] / total_events["carts"]
    recalls["orders"] = total_hits["orders"] / total_events["orders"]

    recalls["avg"] = (
        recalls["clicks"] * weights["clicks"]
        + recalls["carts"] * weights["carts"]
        + recalls["orders"] * weights["orders"]
    )

    return recalls  # , total_hits, total_events


def get_coverage(preds, gts):
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
    preds = df_val[['session', 'candidates', 'pred']].copy()
    preds = preds.sort_values(['session', 'pred'], ascending=[True, False])
    preds = preds[['session', 'candidates', 'pred']].groupby('session').agg(list).reset_index()
    try:
        preds = preds.to_pandas()
    except:
        pass
    preds['candidates'] = preds['candidates'].apply(lambda x: x[:20])

    gt = pd.read_parquet(GT_FILE)
    preds = preds.merge(gt[gt["type"] == target[3:]].drop("type", axis=1), how="left").rename(
        columns={"ground_truth": target}
    )

    n_preds, n_gts, n_found = get_coverage(preds["candidates"].values, preds[target].values)

    if verbose:
        print(f"\n-> {target}  -  Recall : {n_found / n_gts :.4f}\n")
    return n_found / n_gts
