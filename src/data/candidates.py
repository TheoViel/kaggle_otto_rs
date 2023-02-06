import cudf
import glob
import pandas as pd
from collections import Counter

from params import TYPE_LABELS


def load_parquets(regex):
    """
    Loads sessions.

    Args:
        regex (str): Sessions regex

    Returns:
        cudf DataFrame: Sessions.
    """
    dfs = []
    for e, chunk_file in enumerate(glob.glob(regex)):
        chunk = pd.read_parquet(chunk_file)
        chunk.ts = (chunk.ts / 1000).astype("int32")
        chunk["type"] = chunk["type"].map(TYPE_LABELS).astype("int8")
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


def create_candidates(df, clicks_candids, type_weighted_candids, max_cooc=100):
    """
    Theo's algorithm to create candidates from covisitation matrices.
    It uses the top max_cooc ids in the sum of 2 matrices and all the ids in the session

    Args:
        df (cudf DataFrame): Sessions.
        clicks_candids (Candidates): Clicks candidates matrix.
        type_weighted_candids (Candidates): Type weighted candidates matrix.
        max_cooc (int, optional): Maximum coocurence. Defaults to 100.

    Returns:
        pd DataFrame: Candidates.
    """
    df["clicks_candidates"] = df["aid"].map(clicks_candids)
    df["type_weighted_candidates"] = df["aid"].map(type_weighted_candids)

    df["coocurence_candidates"] = (
        df["clicks_candidates"] + df["type_weighted_candidates"]
    )

    df.drop(["clicks_candidates", "type_weighted_candidates"], axis=1, inplace=True)

    df = (
        df.groupby("session")
        .agg({"aid": list, "coocurence_candidates": sum, "type": list})
        .reset_index()
    )

    df["coocurence_candidates"] = df["coocurence_candidates"].parallel_apply(
        lambda x: [aid for aid, _ in Counter(x).most_common(max_cooc)]
        if len(x) > max_cooc
        else x
    )

    df["candidates"] = df["aid"] + df["coocurence_candidates"]
    df["candidates"] = df["candidates"].parallel_apply(lambda x: list(set(x)))
    df.drop(["coocurence_candidates"], axis=1, inplace=True)

    return df


def explode(df, test=False):
    """
    Explodes candidates for saving.

    Args:
        df (cudf DataFrame): Candidates.
        test (bool, optional): Whether data is test data. Defaults to False.

    Returns:
         cudf DataFrame: Exploded candidates.
    """
    if "aid" in df.columns:
        df.drop(["aid", "type"], axis=1, inplace=True)

    df = cudf.from_pandas(df)

    df = df.explode("candidates")
    df["candidates"] = df["candidates"].astype("uint32")
    df["session"] = df["session"].astype("uint32")

    df = df.sort_values(["session", "candidates"]).reset_index(drop=True)

    if not test:
        for col in ["gt_clicks", "gt_carts", "gt_orders"]:
            df_tgt = (
                df[["session", "candidates", col]].explode(col).reset_index(drop=True)
            ).fillna(-1)
            df_tgt[col] = df_tgt[col] == df_tgt["candidates"]
            df_tgt = df_tgt.groupby(["session", "candidates"]).max().reset_index()
            df_tgt = df_tgt.sort_values(["session", "candidates"]).reset_index(
                drop=True
            )

            df[col] = df_tgt[col].astype("uint8")

    return df
