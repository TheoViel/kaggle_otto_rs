import cudf
import itertools
import numpy as np
import pandas as pd
from collections import Counter

from params import TYPE_LABELS


type_weight_multipliers = {0: 1, 1: 6, 2: 3}


def read_file_to_cache(f):
    df = pd.read_parquet(f)
    df.ts = (df.ts / 1000).astype("int32")
    df["type"] = df["type"].map(TYPE_LABELS).astype("int8")
    return df


def suggest_clicks(df, clicks_candids, top_clicks):
    # USE USER HISTORY AIDS AND TYPES
    aids = df.aid.tolist()
    types = df.type.tolist()
    unique_aids = list(dict.fromkeys(aids[::-1]))
    # RERANK CANDIDATES USING WEIGHTS
    if len(unique_aids) >= 20:
        weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = Counter()
        # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
        for aid, w, t in zip(aids, weights, types):
            aids_temp[aid] += w * type_weight_multipliers[t]
        sorted_aids = [k for k, v in aids_temp.most_common(20)]
        return sorted_aids
    # USE "CLICKS" CO-VISITATION MATRIX
    aids2 = list(
        itertools.chain(
            *[clicks_candids[aid] for aid in unique_aids if aid in clicks_candids]
        )
    )
    # RERANK CANDIDATES
    top_aids2 = [
        aid2 for aid2, cnt in Counter(aids2).most_common(20) if aid2 not in unique_aids
    ]
    result = unique_aids + top_aids2[: 20 - len(unique_aids)]
    # USE TOP20 TEST CLICKS
    return result + list(top_clicks)[: 20 - len(result)]


def suggest_buys(df, type_weighted_candids, cartbuy_candids, top_orders):
    # USE USER HISTORY AIDS AND TYPES
    aids = df.aid.tolist()
    types = df.type.tolist()
    # UNIQUE AIDS AND UNIQUE BUYS
    unique_aids = list(dict.fromkeys(aids[::-1]))
    df = df.loc[(df["type"] == 1) | (df["type"] == 2)]
    unique_buys = list(dict.fromkeys(df.aid.tolist()[::-1]))
    # RERANK CANDIDATES USING WEIGHTS
    if len(unique_aids) >= 20:
        weights = np.logspace(0.5, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = Counter()
        # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
        for aid, w, t in zip(aids, weights, types):
            aids_temp[aid] += w * type_weight_multipliers[t]

#         # Increase weight by 0.1 with "BUY2BUY" CO-VISITATION MATRIX
#         aids3 = list(
#             itertools.chain(
#                 *[cartbuy_candids[aid] for aid in unique_buys if aid in cartbuy_candids]
#             )
#         )
#         for aid in aids3:
#             aids_temp[aid] += 0.1

        sorted_aids = [k for k, v in aids_temp.most_common(20)]
        return sorted_aids

    # USE "CART ORDER" CO-VISITATION MATRIX
#     aids2 = list(
#         itertools.chain(
#             *[
#                 type_weighted_candids[aid]
#                 for aid in unique_aids
#                 if aid in type_weighted_candids
#             ]
#         )
#     )
#     # USE "BUY2BUY" CO-VISITATION MATRIX
#     aids3 = list(
#         itertools.chain(
#             *[cartbuy_candids[aid] for aid in unique_buys if aid in cartbuy_candids]
#         )
#     )
#     # New candidates are the top 20 most common in both matrices
#     top_aids2 = [
#         aid2
#         for aid2, cnt in Counter(aids2 + aids3).most_common(20)
#         if aid2 not in unique_aids
#     ]
    result = unique_aids #+ top_aids2[: 20 - len(unique_aids)]
    # USE TOP20 TEST ORDERS
    return result + list(top_orders)[: 20 - len(result)]
