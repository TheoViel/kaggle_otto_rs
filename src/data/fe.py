import os
import gc
import cudf
import numba
import pickle
import numpy as np
from math import log

from params import CLASSES
from utils.load import load_sessions


def compute_weights(sessions, return_sessions=False, no_click=False):
    """
    Computes weights to give each aids in the sessions.

    Args:
        sessions (cudf DataFrame): Sessions.
        return_sessions (bool, optional): Returns the sessions. Defaults to False.
        no_click (bool, optional): Exclude clicks from weighting. Defaults to False.

    Returns:
        cudf DataFrame: Weights.
    """
    sessions = sessions.sort_values(
        ["session", "ts"], ascending=[True, False]
    ).reset_index(drop=True)

    sessions["last_ts"] = sessions.groupby("session")["ts"].cummax()
    sessions["pos"] = sessions.groupby("session")["aid"].cumcount()

    sessions["1"] = 1
    sessions["len"] = sessions.groupby("session")["1"].cumsum()
    sessions["aid_count"] = sessions.groupby("aid")["1"].cumsum()

    sessions["w_pos-log"] = sessions.apply(
        lambda x: 1
        if x.len == 1
        else 2 ** (0.1 + 0.9 * (x.len - x.pos - 1) / (x.len - 1)) - 1,
        axis=1,
    )
    sessions["w_type-163"] = sessions["type"].map({0: 1, 1: 6, 2: 3})
    sessions["w_lastday"] = (
        (sessions["last_ts"] - sessions["ts"]) < 60 * 60 * 24
    ).astype("uint8")

    sessions["dt"] = sessions["last_ts"] - sessions["ts"]

    sessions["w_time"] = sessions["dt"].apply(lambda x: 1 / log(2 + x // 60))
    sessions["w_pos"] = sessions["pos"].apply(lambda x: 1 / log(2 + x))
    sessions["w_sess"] = sessions["len"].apply(lambda x: 1 / log(1 + x))
    sessions["w_aid"] = sessions["aid_count"].apply(lambda x: 1 / log(1 + x))

    # From recsys2022 winners
    sessions["w_recsys"] = (
        sessions["w_time"] * sessions["w_pos"] * sessions["w_sess"] * sessions["w_aid"]
    ).astype("float32")

    sessions.drop(
        ["last_ts", "pos", "1", "len", "aid_count", "w_sess", "w_aid", "w_pos", "dt"],
        axis=1,
        inplace=True,
    )

    for c in sessions.columns[2:]:
        if sessions[c].dtype == "int64":
            sessions[c] = sessions[c].astype("int32")
        elif sessions[c].dtype == "float64":
            sessions[c] = sessions[c].astype("float32")

    if no_click:
        for c in ["w_type-163", "w_lastday", "w_pos-log", "w_time", "w_recsys"]:
            sessions[c] *= sessions["type"] != 0

    if return_sessions:
        return sessions

    weights = (
        sessions.drop(["ts", "type"], axis=1)
        .groupby(["session", "aid"])
        .sum()
        .reset_index()
    )

    weights = (
        weights.sort_values(["session", "aid"])
        .reset_index(drop=True)
        .rename(columns={"aid": "candidates"})
    )

    for c in ["w_type-163", "w_lastday"]:
        weights[c] = weights[c].astype("float32")
    for c in ["w_pos-log", "w_time", "w_recsys"]:
        weights[c] = weights[c].astype("float32")

    return weights


def compute_popularity_features(pairs, parquet_files, suffix=""):
    """
    Computes popularity features.

    Args:
        pairs (cudf DataFrame): Candidates.
        parquet_files (str): Regex to sessions.
        suffix (str, optional): Suffix for feature name. Defaults to "".

    Returns:
        cudf DataFrame: Candidates with features
    """
    sessions = load_sessions(parquet_files)
    sessions = compute_weights(sessions, return_sessions=True)

    offset = len(pairs.columns)
    weightings = list(sessions.columns[4:])

    for i, c in enumerate(CLASSES):
        print(f"-> Popularity for {c} - {suffix}")
        popularity = (
            sessions[sessions["type"] == i][["aid"] + weightings]
            .groupby("aid")
            .sum()
            .reset_index()
        )
        popularity.columns = ["candidates"] + [
            f"{c}_popularity_{w}{suffix}" for w in weightings
        ]
        pairs = pairs.merge(popularity, how="left", on="candidates").fillna(0)

    for c in pairs.columns[offset:]:
        if pairs[c].dtype == "int64":
            pairs[c] = pairs[c].astype("int32")
        elif pairs[c].dtype == "float64":
            pairs[c] = pairs[c].astype("float32")

    del popularity, sessions
    numba.cuda.current_context().deallocations.clear()
    gc.collect()

    return pairs


def compute_popularities_new(pairs, sessions, mode="val"):
    """
    Computes new popularity features.

    Args:
        pairs (cudf DataFrame): Candidates.
        sessions (cudf DataFrame): Sessions.
        mode (str, optional): Suffix for feature name. Defaults to "val".

    Returns:
        cudf DataFrame: Candidates with features
    """
    if mode == "val":
        day_map = {21: 0, 22: 0, 23: 1, 24: 2, 25: 3, 26: 4, 27: 5, 28: 6}  # VAL
    else:
        day_map = {28: 0, 29: 0, 30: 1, 31: 2, 1: 3, 2: 4, 3: 5, 4: 6}  # TEST

    sessions["day"] = (
        cudf.to_datetime(sessions["ts"], unit="s") + cudf.DateOffset(hours=2)
    ).dt.day
    sessions["day"] = sessions["day"].map(day_map)
    sessions["hour"] = (sessions["ts"] // (60 * 60)) * (60 * 60)
    sessions["hour"] = cudf.to_datetime(sessions["hour"], unit="s")
    sessions["count"] = 1

    # Retrieve pairs ts
    pairs = pairs.merge(
        sessions[["session", "ts"]]
        .sort_values(["session", "ts"])
        .groupby("session")
        .agg("last"),
        how="left",
        on="session",
    )

    pairs["day"] = (
        cudf.to_datetime(pairs["ts"], unit="s") + cudf.DateOffset(hours=2)
    ).dt.day
    pairs["day"] = pairs["day"].map(day_map)
    pairs["hour"] = (pairs["ts"] // (60 * 60)) * (60 * 60)
    pairs["hour"] = cudf.to_datetime(pairs["hour"], unit="s")

    for class_idx, c in enumerate(CLASSES):
        #         if class_idx == 0:
        #             print('SKIP CLIC')
        #             continue
        print(f"-> Popularity for {c}")
        # Week
        if os.path.exists(f"../output/popularities/pop_w_{c}_{mode}.parquet"):
            popularity_week = cudf.read_parquet(
                f"../output/popularities/pop_w_{c}_{mode}.parquet"
            )
        else:
            popularity_week = (
                sessions[["aid", "count"]][sessions["type"] == class_idx]
                .groupby("aid")
                .sum()
            )
            popularity_week = popularity_week.reset_index().rename(
                columns={"aid": "candidates"}
            )
            popularity_week[f"popularity_week_{c}"] = (
                popularity_week["count"] * 10000 / popularity_week["count"].sum()
            )
            popularity_week.drop("count", axis=1, inplace=True)

            popularity_week.to_parquet(
                f"../output/popularities/pop_w_{c}_{mode}.parquet"
            )

        # Day
        if os.path.exists(f"../output/popularities/pop_d_{c}_{mode}.parquet"):
            popularity_day = cudf.read_parquet(
                f"../output/popularities/pop_d_{c}_{mode}.parquet"
            )
        else:
            popularity_day = (
                sessions[["aid", "count", "day"]][sessions["type"] == class_idx]
                .groupby(["aid", "day"])
                .sum()
                .reset_index()
            )
            total_day = (
                sessions[["count", "day"]][sessions["type"] == class_idx]
                .groupby(["day"])
                .sum()
                .reset_index()
                .rename(columns={"count": "tot"})
            )
            popularity_day = popularity_day.merge(
                total_day, how="left", on="day"
            ).rename(columns={"aid": "candidates"})

            popularity_day[f"popularity_day_{c}"] = (
                popularity_day["count"] * 10000 / popularity_day["tot"]
            )
            popularity_day.drop(["count", "tot"], axis=1, inplace=True)

            popularity_day.to_parquet(
                f"../output/popularities/pop_d_{c}_{mode}.parquet"
            )

        # Hour  -  Slow for clicks
        if os.path.exists(f"../output/popularities/pop_h_{c}_{mode}.parquet"):
            popularity_hour = cudf.read_parquet(
                f"../output/popularities/pop_h_{c}_{mode}.parquet"
            )
        else:
            popularity_hour = (
                sessions[["aid", "count", "hour"]][sessions["type"] == class_idx]
                .groupby(["aid", "hour"])
                .sum()
                .reset_index()
            )
            popularity_hour = (
                popularity_hour.set_index("hour")
                .sort_index()
                .to_pandas()
                .groupby("aid")
                .rolling("3H", min_periods=1, center=True)
                .sum()
                .reset_index()
            )
            popularity_hour = cudf.from_pandas(popularity_hour)

            total_hour = (
                popularity_hour[["hour", "count"]]
                .groupby(["hour"])
                .sum()
                .reset_index()
                .rename(columns={"count": "tot"})
            )
            popularity_hour = popularity_hour.merge(
                total_hour, how="left", on="hour"
            ).rename(columns={"aid": "candidates"})

            popularity_hour[f"popularity_hour_{c}"] = (
                popularity_hour["count"] * 100 / popularity_hour["tot"]
            )
            popularity_hour.drop(["count", "tot"], axis=1, inplace=True)

            popularity_hour.to_parquet(
                f"../output/popularities/pop_h_{c}_{mode}.parquet"
            )

        # Merge
        pairs = pairs.merge(popularity_week, on="candidates", how="left").fillna(0)
        pairs = pairs.merge(
            popularity_day, on=["candidates", "day"], how="left"
        ).fillna(0)
        pairs = pairs.merge(
            popularity_hour, on=["candidates", "hour"], how="left"
        ).fillna(0)

        pairs[
            [f"popularity_week_{c}", f"popularity_day_{c}", f"popularity_hour_{c}"]
        ] = pairs[
            [f"popularity_week_{c}", f"popularity_day_{c}", f"popularity_hour_{c}"]
        ].astype(
            "float32"
        )
        pairs[f"popularity_hour/day_{c}"] = (
            pairs[f"popularity_hour_{c}"] / pairs[f"popularity_day_{c}"]
        ).fillna(0)
        pairs[f"popularity_day/week_{c}"] = (
            pairs[f"popularity_day_{c}"] / pairs[f"popularity_week_{c}"]
        ).fillna(0)

    pairs.drop(["ts", "day", "hour"], axis=1, inplace=True)
    return pairs


def compute_coocurence_features(pairs, matrix_file, weights):
    """
    Computes features from coocurence / covisitation matrices.

    Args:
        pairs (cudf DataFrame): Candidates.
        matrix_file (str): Path to matrix.
        weights (cudf DataFrame): Weights for aggregation.

    Returns:
        cudf DataFrame: Candidates with features.
    """
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
    """
    Actions count features.

    Args:
        pairs (cudf DataFrame): Candidates.
        sessions (cudf DataFrame): Sessions.

    Returns:
        np array: Action count feature.
    """
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
    """
    Adds the groupby rank feature.
    Implementation can be simplified.

    Args:
        pairs (cudf DataFrame): Candidates.
        feature (str): Feature name.
    """
    df_ft = pairs[["session", "candidates", feature]]
    df_ft = df_ft.sort_values(feature, ascending=False, ignore_index=True)
    df_ft[f"{feature}_rank"] = 1
    df_ft[f"{feature}_rank"] = df_ft[f"{feature}_rank"].astype("uint8")
    df_ft[f"{feature}_rank"] = df_ft.groupby("session")[f"{feature}_rank"].cumsum()

    df_ft[f"{feature}_rank"] = df_ft.groupby(["session", feature])[
        f"{feature}_rank"
    ].cummin()  # Ties

    df_ft = df_ft.drop(feature, axis=1).sort_values(
        ["session", "candidates"], ignore_index=True
    )
    pairs[f"{feature}_rank"] = np.clip(df_ft[f"{feature}_rank"], 0, 255).astype("uint8")


def compute_matrix_factorization_features(pairs, embed, weights):
    """
    Computes features from matrix factorization embeddings.

    Args:
        pairs (cudf DataFrame): Candidates.
        embed (np array): Embeddings.
        weights (cudf DataFrame): Weights for aggreagtion.

    Returns:
        cudf DataFrame: Features.
    """
    pairs["group"] = pairs["session"] // 100000

    weights = weights.rename(columns={"candidates": "aid"})
    weightings = list(weights.columns[2:])

    fts = []
    for _, df in pairs.groupby("group"):
        df = df[["session", "candidates", "aid"]].explode("aid").reset_index(drop=True)

        df["w"] = np.sum(
            embed[df["candidates"].to_pandas().values]
            * embed[df["aid"].to_pandas().values],
            axis=1,
        )
        df["w"] += df["aid"] != embed.shape[0] - 1  # non-nan are put in [0, 2]

        df = df.merge(weights, how="left", on=["session", "aid"])

        for weighting in weightings:
            df[weighting] *= df["w"]

        df = (
            df[["candidates", "session"] + weightings]
            .groupby(["session", "candidates"])
            .agg(["mean", "sum", "max"])
        )
        df.columns = ["_".join(col) for col in df.columns.values]

        df[df.columns] = df[df.columns].astype("float32")
        fts.append(df.reset_index())

    fts = cudf.concat(fts, ignore_index=True)
    fts = fts.sort_values(["session", "candidates"]).reset_index(drop=True)

    return fts


def benny_weights(df):
    """
    Computes Benny's weights.

    Args:
        df (cudf DataFrame): Sessions.

    Returns:
        cudf DataFrame: Weights.
    """
    df_type = cudf.DataFrame(
        {"type": [0, 1, 2], "type_": [1, 2, 3], "type_mp": [0.5, 9, 0.5]}
    )

    df = df.merge(df_type, how="left", on="type")
    df = df.sort_values(["session", "ts"], ascending=[True, False])

    df["session_len"] = df.groupby("session").transform("count")["aid"]

    df["rank"] = 1
    df["rank"] = df.groupby(["session"])["rank"].cumsum().astype("int32")
    df["last"] = (df["rank"] == 1).astype("uint8")

    min_val = 2**0.1 - 1
    max_val = 2**1 - 1
    df["wgt_1"] = (
        min_val + (max_val - min_val) * (df["rank"] - 1) / df["session_len"]
    ) * df["type_mp"]
    min_val = 2**0.5 - 1
    max_val = 2**1 - 1
    df["wgt_2"] = (
        min_val + (max_val - min_val) * (df["rank"] - 1) / df["session_len"]
    ) * df["type_mp"]

    df["wgt_1"] = df["wgt_1"].astype("float32")
    df["wgt_2"] = df["wgt_2"].astype("float32")
    df["type_mp"] = df["type_mp"].astype("float32")

    return df.drop(["type_", "session_len"], axis=1)


def load_embed(embed_file):
    """
    Loads and normalizes an embedding matrix.

    Args:
        embed_file (str): Path to embeddings.

    Returns:
        np array: Embeddings.
    """
    if not embed_file.endswith(".npy"):
        emb = pickle.load(open(embed_file, "rb"))
        embed = np.zeros((np.max(list(emb.keys())) + 1, 50), dtype=np.float32)
        for k in emb.keys():
            embed[k] = emb[k]
    else:
        embed = np.load(embed_file)

    embed /= np.reshape(np.sqrt(np.sum(embed * embed, axis=1)), (-1, 1)) + 1e-12
    embed = np.concatenate((embed, np.zeros((1, embed.shape[1])))).astype(np.float32)
    return embed


def compute_w2v_features(pairs, parquet_files, embed, name="w2v"):
    """
    Computes Benny's Word2vec features.
    Also works with other embeddings.

    Args:
        pairs (cudf DataFrame): Candidates.
        parquet_files (str): Regex to sessions.
        embed (np array): Embedding matrix.
        name (str, optional): Matrix name. Defaults to "w2v".

    Returns:
        cudf DataFrame: Features.
    """
    pairs = pairs.sort_values(["session", "candidates"], ignore_index=True)

    sessions = load_sessions(parquet_files)
    sessions = benny_weights(sessions)

    df_pairs = pairs[["session", "candidates"]].merge(
        sessions, how="left", on="session"
    )

    df_pairs["sim"] = np.sum(
        embed[df_pairs["candidates"].to_pandas().values]
        * embed[df_pairs["aid"].to_pandas().values],
        axis=1,
    )
    df_pairs.loc[df_pairs["sim"] < 0.5, "sim"] = 0

    df_pairs["sim_1"] = df_pairs["sim"].values
    df_pairs["sim_2"] = df_pairs["sim"].values
    df_pairs["sim_3"] = (df_pairs["sim"] > 0).astype("int")
    df_pairs["sim_wgt_1"] = df_pairs["sim"] * df_pairs["wgt_1"]
    df_pairs["sim_wgt_2"] = df_pairs["sim"] * df_pairs["wgt_2"]
    df_pairs["sim_last"] = df_pairs["sim"] * df_pairs["last"]
    df_pairs["sim_type_1"] = df_pairs["sim"] * df_pairs["type_mp"]

    df_pairs = df_pairs.groupby(["session", "candidates"]).agg(
        {
            "sim_1": "max",
            "sim_2": "sum",
            "sim_3": "sum",
            "sim_wgt_1": "sum",
            "sim_wgt_2": "sum",
            "sim_last": "max",
            "sim_type_1": "sum",
        }
    )
    for col in ["sim_2", "sim_wgt_1", "sim_wgt_2", "sim_type_1"]:
        df_pairs[col] = df_pairs[col] / (1 + df_pairs["sim_3"])

    df_pairs = df_pairs.reset_index().sort_values(
        ["session", "candidates"], ignore_index=True
    )
    assert (pairs["candidates"] == df_pairs["candidates"]).all()
    df_pairs.drop(["session", "candidates"], axis=1, inplace=True)

    for c in df_pairs.columns:
        if "sim_3" in c:
            df_pairs[c] = df_pairs[c].astype("int32")
        else:
            df_pairs[c] = df_pairs[c].astype("float32")

    df_pairs.columns = [f"{name}_{c}" for c in list(df_pairs.columns)]
    pairs = cudf.concat([pairs, df_pairs], axis=1)

    return pairs


def save_by_chunks(pairs, folder, part=0, chunk_size=50000):
    """
    Saves the computed features by chunk.

    Args:
        pairs (cudf DataFrame): Features.
        folder (str): Folder to save into.
        part (int, optional): Feature part index.. Defaults to 0.
        chunk_size (int, optional): Chunk size. Defaults to 50000.
    """
    print(f"-> Saving chunks to {folder}   (part #{part})")
    os.makedirs(folder, exist_ok=True)

    pairs["group"] = pairs["session"] // chunk_size

    for i, (_, df) in enumerate(pairs.groupby("group")):
        df.drop("group", axis=1, inplace=True)
        df.to_pandas().to_parquet(os.path.join(folder, f"{part}_{i:04d}.parquet"))
