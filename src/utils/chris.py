import os
import gc
import cudf
import glob
import itertools
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import Counter


type_weight_multipliers = {0: 1, 1: 6, 2: 3}
type_labels = {"clicks": 0, "carts": 1, "orders": 2}


def read_file(f, data_cache):
    return cudf.DataFrame(data_cache[f])


def read_file_to_cache(f):
    df = pd.read_parquet(f)
    df.ts = (df.ts / 1000).astype("int32")
    df["type"] = df["type"].map(type_labels).astype("int8")
    return df


def load_val():
    dfs = []
    for e, chunk_file in enumerate(glob.glob("../input/chris/test_parquet/*")):
        chunk = pd.read_parquet(chunk_file)
        chunk.ts = (chunk.ts / 1000).astype("int32")
        chunk["type"] = chunk["type"].map(type_labels).astype("int8")
        dfs.append(chunk)
    return pd.concat(dfs).reset_index(drop=True)



class Candidates(dict):
    def __missing__(self, key):
        return []


def matrix_to_candids_dict(matrix):
    candids = matrix[["aid_x", "aid_y"]].groupby("aid_x").agg(list)

    try:
        candids = candids.to_pandas()
    except AttributeError:
        pass

    candids["aid_y"] = candids["aid_y"].apply(lambda x: x.tolist())
    candids_dict = candids.to_dict()["aid_y"]

    candids_dict = Candidates(candids_dict)

    return candids_dict


def compute_covisitation_matrix(
    files,
    data_cache,
    weighting="",
    type_weight={},
    considered_types=[1, 2, 3],
    n=0,
    save_folder="",
):
    # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS => OUTER CHUNKS
    for j in tqdm(range(6)):
        READ_CT = 6
        CHUNK = int(np.ceil(len(files) / READ_CT))

        a = j * CHUNK
        b = min((j + 1) * CHUNK, len(files))

        # => INNER CHUNKS
        for k in range(a, b, READ_CT):
            # READ FILE
            df = [read_file(files[k], data_cache)]
            for i in range(1, READ_CT):
                if k + i < b:
                    df.append(read_file(files[k + i], data_cache))
            df = cudf.concat(df, ignore_index=True, axis=0)

            if considered_types != [1, 2, 3]:
                df = df.loc[df["type"].isin(considered_types)]

            df = df.sort_values(["session", "ts"], ascending=[True, False])

            # USE TAIL OF SESSION
            df = df.reset_index(drop=True)
            df["n"] = df.groupby("session").cumcount()
            df = df.loc[df.n < 30].drop("n", axis=1)

            # CREATE PAIRS
            df = df.merge(df, on="session")
            df = df.loc[  # Less than 1h appart, different ID
                ((df.ts_x - df.ts_y).abs() < 24 * 60 * 60) & (df.aid_x != df.aid_y)
            ]

            # ASSIGN WEIGHTS
            df = df[["session", "aid_x", "aid_y", "ts_x", "type_y"]].drop_duplicates(
                ["session", "aid_x", "aid_y"]
            )  # duplicates

            if weighting == "temporal":
                df.drop("type_y", axis=1, inplace=True)
                df["wgt"] = 1 + 3 * (df.ts_x - 1659304800) / (1662328791 - 1659304800)
            elif weighting == "type":
                df.drop("ts_x", axis=1, inplace=True)
                df["wgt"] = df.type_y.map(type_weight)
            else:
                df.drop(["type_y", "ts_x"], axis=1, inplace=True)
                df["wgt"] = 1

            df = df[["aid_x", "aid_y", "wgt"]]
            df.wgt = df.wgt.astype("float32")
            df = df.groupby(["aid_x", "aid_y"]).wgt.sum()

            # COMBINE INNER CHUNKS
            if k == a:
                matrix_chunk = df
            else:
                matrix_chunk = matrix_chunk.add(df, fill_value=0)

        # COMBINE OUTER CHUNKS
        if a == 0:
            matrix = matrix_chunk
        else:
            matrix = matrix.add(matrix_chunk, fill_value=0)
        del matrix_chunk, df
        gc.collect()

    # CONVERT MATRIX TO DICTIONARY
    matrix = matrix.reset_index()
    matrix = matrix.sort_values(["aid_x", "wgt"], ascending=[True, False])

    # SAVE TOP 40
    matrix = matrix.reset_index(drop=True)
    matrix["n"] = matrix.groupby("aid_x").aid_y.cumcount()

    if n:
        matrix = matrix.loc[matrix.n < n].drop("n", axis=1)

    if save_folder:
        if weighting == "type":
            weighting += "".join(map(str, list(type_weight.values())))
        save_path = os.path.join(
            save_folder,
            f'matrix_{"".join(map(str, considered_types))}_{weighting}_{n}.pqt',
        )
        print(f"Saving matrix to {save_path}")
        matrix.to_pandas().to_parquet(save_path)

    candids = matrix_to_candids_dict(matrix)

    return candids


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
        # RERANK CANDIDATES USING "BUY2BUY" CO-VISITATION MATRIX
        aids3 = list(
            itertools.chain(
                *[
                    cartbuy_candids[aid]
                    for aid in unique_buys
                    if aid in cartbuy_candids
                ]
            )
        )
        for aid in aids3:
            aids_temp[aid] += 0.1
        sorted_aids = [k for k, v in aids_temp.most_common(20)]
        return sorted_aids
    # USE "CART ORDER" CO-VISITATION MATRIX
    aids2 = list(
        itertools.chain(
            *[
                type_weighted_candids[aid]
                for aid in unique_aids
                if aid in type_weighted_candids
            ]
        )
    )
    # USE "BUY2BUY" CO-VISITATION MATRIX
    aids3 = list(
        itertools.chain(
            *[
                cartbuy_candids[aid]
                for aid in unique_buys
                if aid in cartbuy_candids
            ]
        )
    )
    # RERANK CANDIDATES
    top_aids2 = [
        aid2
        for aid2, cnt in Counter(aids2 + aids3).most_common(20)
        if aid2 not in unique_aids
    ]
    result = unique_aids + top_aids2[: 20 - len(unique_aids)]
    # USE TOP20 TEST ORDERS
    return result + list(top_orders)[: 20 - len(result)]