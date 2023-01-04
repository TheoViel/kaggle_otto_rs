import os
import gc
import cudf
import numba
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

from params import CLASSES
from utils.load import load_sessions


def compute_weights(sessions):
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
    sessions["linspace_w"] = sessions["w"].apply(
        lambda x: 0.05 if x >= 20 else 0.1 + 0.9 * (18 - x) / 18
    )

    sessions["linspace_w_t163"] = sessions["linspace_w"] * sessions["type"].map(
        {0: 1, 1: 6, 2: 3}
    )
    sessions["logspace_w_t163"] = sessions["logspace_w"] * sessions["type"].map(
        {0: 1, 1: 6, 2: 3}
    )

    sessions["linspace_w_t191"] = sessions["linspace_w"] * sessions["type"].map(
        {0: 1, 1: 9, 2: 1}
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
        
        
def compute_popularity_features(pairs, parquet_files, suffix):
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


def compute_popularity_features_2(pairs, parquet_files, suffix):
    sessions = load_sessions(parquet_files)

    sessions["w"] = sessions["ts"] - sessions["ts"].min()
    max_ = sessions["w"].max()
    sessions["w"] = sessions["w"].apply(
        lambda x: 2 ** (0.1 + 0.9 * (x - 1) / (max_ - 1)) - 1
    )

    for i, c in enumerate(CLASSES):
        print(f"-> Popularity for {c} - {suffix}")
        popularity = cudf.DataFrame(
            sessions.loc[sessions["type"] == i, "aid"].value_counts()
        ).reset_index()
        popularity.columns = ["candidates", f"{c}_popularity{suffix}"]
        popularity[f"{c}_popularity{suffix}"] = np.clip(
            popularity[f"{c}_popularity{suffix}"], 0, 2**31 - 1
        ).astype("int32")

        pairs = pairs.merge(popularity, how="left", on="candidates").fillna(0)

        popularity_time_weighted = (
            sessions[sessions["type"] == i][["aid", "w"]]
            .groupby("aid")
            .sum()
            .reset_index()
        )
        popularity_time_weighted["w"] = popularity_time_weighted["w"].astype("float32")

        popularity_time_weighted.columns = [
            "candidates",
            f"{c}_popularity_log{suffix}",
        ]
        pairs = pairs.merge(
            popularity_time_weighted, how="left", on="candidates"
        ).fillna(0)

        del popularity, popularity_time_weighted
        numba.cuda.current_context().deallocations.clear()
        gc.collect()

    print(f"-> Popularity for views - {suffix}")
    popularity = cudf.DataFrame(
        sessions["aid"].value_counts()
    ).reset_index()
    popularity.columns = ["candidates", f"views_popularity{suffix}"]
    popularity[f"views_popularity{suffix}"] = np.clip(
        popularity[f"views_popularity{suffix}"], 0, 2**31 - 1
    ).astype("int32")

    pairs = pairs.merge(popularity, how="left", on="candidates").fillna(0)

    popularity_time_weighted = sessions[["aid", "w", "w_log"]].groupby("aid").sum().reset_index()
    popularity_time_weighted["w"] = popularity_time_weighted["w"].astype("float32")
    popularity_time_weighted["w_log"] = popularity_time_weighted["w_log"].astype("float32")

    popularity_time_weighted.columns = ["candidates", f"views_popularity_log{suffix}"]
    pairs = pairs.merge(
        popularity_time_weighted, how="left", on="candidates"
    ).fillna(0)

    del popularity, popularity_time_weighted
    numba.cuda.current_context().deallocations.clear()
    gc.collect()

    return pairs


def compute_coocurence_features(pairs, matrix_file, weights):
    pairs["group"] = pairs["session"] // 100000

    weights = weights.rename(columns={"candidates": "aid"})

    mat = cudf.read_parquet(matrix_file)
    mat.columns = ["aid", "candidates", "w"]

    fts = []
    for _, df in tqdm(pairs.groupby("group")):
        df = df[["session", "candidates", "aid"]].explode("aid").reset_index(drop=True)

        df = df.merge(mat, how="left", on=["aid", "candidates"]).reset_index().fillna(0)

        df = df.merge(weights, how="left", on=["session", "aid"])
        df["logspace_w"] *= df["w"]
        df["linspace_w"] *= df["w"]

        df = (
            df[["candidates", "session", "w", "logspace_w", "linspace_w"]]
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
    for _, df in tqdm(pairs.groupby("group")):
        df = df[["session", "candidates", "aid"]].explode("aid")
        df["aid"] = (df["aid"] == df["candidates"]).astype(np.uint16)

        df = df.groupby(["session", "candidates"]).sum().reset_index()

        fts.append(df)

    ft = cudf.concat(fts, ignore_index=True)
    ft = ft.sort_values(["session", "candidates"])["aid"].values

    return np.clip(ft, 0, 255).astype(np.uint8)


def get_time(x):
    return datetime.utcfromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S")


def get_period(x):
    if 7 <= x <= 12:
        return 0  # "morning"
    elif 12 <= x <= 18:
        return 1  # "afernoon"
    elif 18 <= x <= 23:
        return 2  # "evening"
    else:
        return 3  # "night"


def get_datetime_info(df):
    df["datetime"] = cudf.to_datetime(
        df["ts"].to_pandas().parallel_apply(get_time).values
    )
    df["datetime"] += cudf.DateOffset(hours=2)  # UTC + 2 in germany

    df["day"] = df["datetime"].dt.day + (df["datetime"].dt.month - 8) * 12
    df["weekday"] = df["datetime"].dt.weekday
    df["hour"] = df["datetime"].dt.hour
    #     df["period"] = df["hour"].apply(get_period)

    df[["day", "hour", "weekday"]] = df[["day", "hour", "weekday"]].astype("uint8")
    #     df[["day", "hour", "weekday", "period"]] = df[["day", "hour", "weekday", "period"]].astype("uint8")
    df.drop("datetime", axis=1, inplace=True)
    return df


def save_by_chunks(pairs, folder, part=0):
    print(f"-> Saving chunks to {folder}   (part #{part})")
    os.makedirs(folder, exist_ok=True)

    pairs["group"] = pairs["session"] // 100000

    for i, (_, df) in enumerate(tqdm(pairs.groupby("group"))):
        df.drop("group", axis=1, inplace=True)
        df.to_parquet(os.path.join(folder, f"{part}_{i:03d}.parquet"))
