import os
import gc
import cudf
import numba
import datetime
import numpy as np
import pandas as pd
from math import log

from params import CLASSES
from utils.load import load_sessions


def compute_weights_old(sessions):
    sessions.sort_values(["session", "ts"], ascending=[True, False]).reset_index(
        drop=True
    )

    sessions["w"] = sessions.groupby("session")["aid"].cumcount()

    sessions = sessions.merge(
        cudf.DataFrame(sessions.groupby("session")["aid"].size()),
        on="session",
        how="left",
    ).rename(columns={0: "n"})

    sessions["logspace_w"] = sessions.apply(
        lambda x: 1 if x.n == 1 else 2 ** (0.1 + 0.9 * (x.n - x.w - 1) / (x.n - 1)) - 1,
        axis=1,
    )
    sessions["logspace_w_t163"] = sessions["logspace_w"] * sessions["type"].map(
        {0: 1, 1: 6, 2: 3}
    )
    sessions["logspace_w_t191"] = sessions["logspace_w"] * sessions["type"].map(
        {0: 1, 1: 9, 2: 1}
    )

    weights = (
        sessions.drop(["ts", "type", "w", "n"], axis=1)
        .groupby(["session", "aid"])
        .sum()
        .reset_index()
    )

    weights = (
        weights.sort_values(["session", "aid"])
        .reset_index(drop=True)
        .rename(columns={"aid": "candidates"})
    )

    return weights
        
    
def compute_weights(sessions, return_sessions=False):
    sessions = sessions.sort_values(["session", "ts"], ascending=[True, False]).reset_index(
        drop=True
    )

    sessions["last_ts"] = sessions.groupby("session")["ts"].cummax()
#     sessions["first_ts"] = sessions.groupby("session")["ts"].cummin()
    sessions["pos"] = sessions.groupby("session")["aid"].cumcount()

    sessions['1'] = 1
    sessions["len"] = sessions.groupby("session")["1"].cumsum()
    sessions["aid_count"] = sessions.groupby("aid")["1"].cumsum()

    sessions["w_pos-log"] = sessions.apply(
        lambda x: 1 if x.len == 1 else 2 ** (0.1 + 0.9 * (x.len - x.pos - 1) / (x.len - 1)) - 1,
        axis=1,
    )
    sessions["w_type-163"] = sessions["type"].map({0: 1, 1: 6, 2: 3})
#     sessions["w_type-191"] = sessions["type"].map({0: 1, 1: 9, 2: 1})
    
#     sessions["w_lastmin"] = ((sessions["last_ts"] - sessions["ts"]) < 60).astype("uint8")
#     sessions["w_lasthour"] = ((sessions["last_ts"] - sessions["ts"]) < 60 * 60).astype("uint8")
    sessions["w_lastday"] = ((sessions["last_ts"] - sessions["ts"]) < 60 * 60 * 24).astype("uint8")

    sessions["dt"] = (sessions["last_ts"] - sessions["ts"])

    sessions["w_time"] = sessions["dt"].apply(lambda x: 1 / log(2 + x // 60))
    sessions["w_pos"] = sessions["pos"].apply(lambda x: 1 / log(2 + x))
    sessions["w_sess"] = sessions["len"].apply(lambda x: 1 / log(1 + x))
    sessions["w_aid"] = sessions["aid_count"].apply(lambda x: 1 / log(1 + x))

    sessions['w_recsys'] = (sessions["w_time"] * sessions["w_pos"] * sessions["w_sess"] * sessions["w_aid"]).astype("float32")

    sessions.drop(['last_ts', 'pos', '1', 'len', 'aid_count', 'w_sess', 'w_aid', 'w_pos', 'dt'], axis=1, inplace=True)
    
    for c in sessions.columns[2:]:
        if sessions[c].dtype == "int64":
            sessions[c] = sessions[c].astype('int32')
        elif sessions[c].dtype == "float64":
            sessions[c] = sessions[c].astype('float32')
            
    if return_sessions:
        return sessions

    weights = sessions.drop(["ts", "type"], axis=1).groupby(["session", "aid"]).sum().reset_index()

    weights = (
        weights.sort_values(["session", "aid"])
        .reset_index(drop=True)
        .rename(columns={"aid": "candidates"})
    )

    for c in ["w_type-163", "w_lastday"]:  # "w_type-191", "w_lastmin", "w_lasthour", 
#         weights[c] = np.clip(weights[c], 0, 127).astype("int8")
        weights[c] = weights[c].astype("float32")
    for c in ["w_pos-log", "w_time", "w_recsys"]:
        weights[c] = weights[c].astype("float32")

    return weights


def compute_popularity_features_old(pairs, parquet_files, suffix):
    """
    TODO : More popularities using weights!
    """
    sessions = load_sessions(parquet_files)

    sessions["w"] = sessions["ts"] - sessions["ts"].min()
    max_ = sessions["w"].max()
    sessions["w_log"] = sessions["w"].apply(
        lambda x: 2 ** (0.1 + 0.9 * (x - 1) / (max_ - 1)) - 1
    )
    sessions["w"] = sessions["w"].apply(lambda x: 0.1 + 0.9 * (x - 1) / (max_ - 1))

    for i, c in enumerate(CLASSES):
        print(f"-> Popularity for {c} - {suffix}")
        popularity = cudf.DataFrame(
            sessions.loc[sessions["type"] == i, "aid"].value_counts()
        ).reset_index()
        popularity.columns = ["candidates", f"{c}_popularity{suffix}"]
        popularity[f"{c}_popularity{suffix}"] = np.clip(
            popularity[f"{c}_popularity{suffix}"], 0, 2**16 - 1
        ).astype("uint16")

        pairs = pairs.merge(popularity, how="left", on="candidates").fillna(0)

    popularity_time_weighted = sessions[["aid", "w", "w_log"]].groupby("aid").sum().reset_index()
    popularity_time_weighted["w"] = popularity_time_weighted["w"].astype("float32")
    popularity_time_weighted["w_log"] = popularity_time_weighted["w_log"].astype("float32")
    popularity_time_weighted.columns = ["candidates", f"view_popularity_lin{suffix}", f"view_popularity_log{suffix}"]

    pairs = pairs.merge(
        popularity_time_weighted, how="left", on="candidates"
    ).fillna(0)

    del popularity, popularity_time_weighted, sessions
    numba.cuda.current_context().deallocations.clear()
    gc.collect()

    return pairs


def compute_popularity_features(pairs, parquet_files, suffix=""):
    sessions = load_sessions(parquet_files)    
    sessions = compute_weights(sessions, return_sessions=True)
    
    offset = len(pairs.columns)
    weightings = list(sessions.columns[4:])

    for i, c in enumerate(CLASSES):
        print(f"-> Popularity for {c} - {suffix}")
        popularity = sessions[sessions["type"] == i][["aid"] + weightings].groupby("aid").sum().reset_index()
        popularity.columns = ["candidates"] + [f"{c}_popularity_{w}{suffix}" for w in weightings]
        pairs = pairs.merge(popularity, how="left", on="candidates").fillna(0)

#     print(f"-> Popularity for views - {suffix}")
#     popularity = sessions[["aid"] + weightings].groupby("aid").sum().reset_index()
#     popularity.columns = ["candidates"] + [f"popularity_{w}_{suffix}" for w in weightings]
#     pairs = pairs.merge(popularity, how="left", on="candidates").fillna(0)

    for c in pairs.columns[offset:]:
        if pairs[c].dtype == "int64":
            pairs[c] = pairs[c].astype('int32')
        elif pairs[c].dtype == "float64":
            pairs[c] = pairs[c].astype('float32')

    del popularity, sessions
    numba.cuda.current_context().deallocations.clear()
    gc.collect()

    return pairs


def compute_coocurence_features(pairs, matrix_file, weights):
    pairs["group"] = pairs["session"] // 100000

    weights = weights.rename(columns={"candidates": "aid"})
    weightings = list(weights.columns[2:])

    mat = cudf.read_parquet(matrix_file)
    mat.columns = ["aid", "candidates", "w"]

    fts = []
    for _, df in pairs.groupby("group"):
        df = df[["session", "candidates", "aid"]].explode("aid").reset_index(drop=True)

        df = df.merge(mat, how="left", on=["aid", "candidates"]).reset_index().fillna(0)

        df = df.merge(weights, how="left", on=["session", "aid"])
        
        for weighting in weightings:
            df[weighting] *= df["w"]

        df = (
            df[["candidates", "session", "w"] + weightings]
            .groupby(["session", "candidates"])
            .agg(["mean", "sum", "max"])
        )
        df.columns = ["_".join(col) for col in df.columns.values]

        df[df.columns] = df[df.columns].astype("float32")
        fts.append(df.reset_index())

    fts = cudf.concat(fts, ignore_index=True)
    fts = fts.sort_values(["session", "candidates"]).reset_index(drop=True)
    return fts


def count_actions(pairs, sessions):
    pairs = pairs.merge(sessions[["session", "aid"]], how="left", on="session")
    pairs["group"] = pairs["session"] // 100000

    fts = []
    for _, df in pairs.groupby("group"):
        df = df[["session", "candidates", "aid"]].explode("aid")
        df["aid"] = (df["aid"] == df["candidates"]).astype(np.uint16)

        df = df.groupby(["session", "candidates"]).sum().reset_index()

        fts.append(df)

    ft = cudf.concat(fts, ignore_index=True)
    ft = ft.sort_values(["session", "candidates"])["aid"].values

    return np.clip(ft, 0, 255).astype(np.uint8)


def add_rank_feature(pairs, feature):
    df_ft = pairs[["session", "candidates", feature]]
    df_ft = df_ft.sort_values(feature, ascending=False, ignore_index=True)
    df_ft[f'{feature}_rank'] = 1
    df_ft[f'{feature}_rank'] = df_ft[f'{feature}_rank'].astype("uint8")
    df_ft[f'{feature}_rank'] = df_ft.groupby("session")[f'{feature}_rank'].cumsum()

    df_ft[f'{feature}_rank'] = df_ft.groupby(["session", feature])[f'{feature}_rank'].cummin()  # Ties

    df_ft = df_ft.drop(feature, axis=1).sort_values(["session", "candidates"], ignore_index=True)
    
    pairs[f'{feature}_rank'] = df_ft[f'{feature}_rank'].astype("uint8")


def save_by_chunks(pairs, folder, part=0):
    print(f"-> Saving chunks to {folder}   (part #{part})")
    os.makedirs(folder, exist_ok=True)

    pairs["group"] = pairs["session"] // 100000

    for i, (_, df) in enumerate(pairs.groupby("group")):
        df.drop("group", axis=1, inplace=True)
        df.to_parquet(os.path.join(folder, f"{part}_{i:03d}.parquet"))
