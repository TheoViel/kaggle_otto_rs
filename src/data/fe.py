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
        
    
def compute_weights(sessions, return_sessions=False, no_click=False):
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

    # From recsys2022 winners
    sessions['w_recsys'] = (sessions["w_time"] * sessions["w_pos"] * sessions["w_sess"] * sessions["w_aid"]).astype("float32")

    sessions.drop(['last_ts', 'pos', '1', 'len', 'aid_count', 'w_sess', 'w_aid', 'w_pos', 'dt'], axis=1, inplace=True)

    for c in sessions.columns[2:]:
        if sessions[c].dtype == "int64":
            sessions[c] = sessions[c].astype('int32')
        elif sessions[c].dtype == "float64":
            sessions[c] = sessions[c].astype('float32')

    if no_click:
        for c in ["w_type-163", "w_lastday", "w_pos-log", "w_time", "w_recsys"]:
            sessions[c] *= (sessions["type"] != 0)

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

    for c in pairs.columns[offset:]:
        if pairs[c].dtype == "int64":
            pairs[c] = pairs[c].astype('int32')
        elif pairs[c].dtype == "float64":
            pairs[c] = pairs[c].astype('float32')

    del popularity, sessions
    numba.cuda.current_context().deallocations.clear()
    gc.collect()

    return pairs


def compute_popularities_new(pairs, sessions, mode="val"):
    if mode == "val":
        day_map = {21: 0, 22: 0, 23: 1, 24: 2, 25: 3, 26: 4, 27: 5, 28: 6}  # VAL
    else:
        day_map = {28: 0, 29: 0, 30: 1, 31: 2, 1: 3, 2: 4, 3: 5, 4: 6}  # TEST
    
    sessions["day"] = cudf.to_datetime(sessions['ts'], unit='s').dt.day
    sessions["day"] = sessions["day"].map(day_map)
    sessions["hour"] = (sessions['ts'] // (60 * 60)) * (60 * 60)
    sessions["hour"] = cudf.to_datetime(sessions['hour'], unit='s')
    sessions["count"] = 1
    
    # Retrieve pairs ts
    pairs = pairs.merge(
        sessions[['session', 'ts']].sort_values(['session', 'ts']).groupby('session').agg('last'),
        how="left",
        on="session",
    )

    pairs["day"] = cudf.to_datetime(pairs['ts'], unit='s').dt.day
    pairs["day"] = pairs["day"].map(day_map)
    pairs["hour"] = (pairs['ts'] // (60 * 60)) * (60 * 60)
    pairs["hour"] = cudf.to_datetime(pairs['hour'], unit='s')

    for class_idx, c in enumerate(CLASSES):
        print(f"-> Popularity for {c}")
        # Week        
        if os.path.exists(f"../output/popularities/pop_d_{c}_{mode}.parquet"):
            popularity_week = cudf.read_parquet(f"../output/popularities/pop_w_{c}_{mode}.parquet")
        else:
            popularity_week = sessions[["aid", "count"]][sessions["type"] == class_idx].groupby("aid").sum()
            popularity_week = popularity_week.reset_index().rename(columns={"aid": "candidates"})
            popularity_week[f"popularity_week_{c}"] = popularity_week['count'] * 10000 / popularity_week["count"].sum()
            popularity_week.drop("count", axis=1, inplace=True)
        
            popularity_week.to_parquet(f"../output/popularities/pop_w_{c}_{mode}.parquet")

        # Day    
        if os.path.exists(f"../output/popularities/pop_d_{c}_{mode}.parquet"):
            popularity_day = cudf.read_parquet(f"../output/popularities/pop_d_{c}_{mode}.parquet")
        else:
            popularity_day = sessions[["aid", "count", "day"]][sessions["type"] == class_idx].groupby(["aid", "day"]).sum().reset_index()
            total_day = sessions[["count", "day"]][sessions["type"] == class_idx].groupby(["day"]).sum().reset_index().rename(columns={"count": "tot"})
            popularity_day = popularity_day.merge(total_day, how="left", on="day").rename(columns={"aid": "candidates"})

            popularity_day[f'popularity_day_{c}'] = popularity_day['count'] * 10000 / popularity_day['tot']
            popularity_day.drop(["count", "tot"], axis=1, inplace=True)
        
            popularity_day.to_parquet(f"../output/popularities/pop_d_{c}_{mode}.parquet")

        # Hour  -  Slow for clicks
        if os.path.exists(f"../output/popularities/pop_h_{c}_{mode}.parquet"):
            popularity_hour = cudf.read_parquet(f"../output/popularities/pop_h_{c}_{mode}.parquet")
        else:
            popularity_hour = sessions[["aid", "count", "hour"]][sessions["type"] == class_idx].groupby(["aid", "hour"]).sum().reset_index()
            popularity_hour = popularity_hour.set_index('hour').sort_index().to_pandas().groupby('aid').rolling("3H", min_periods=1, center=True).sum().reset_index()
            popularity_hour = cudf.from_pandas(popularity_hour)

            total_hour = popularity_hour[["hour", "count"]].groupby(["hour"]).sum().reset_index().rename(columns={"count": "tot"})
            popularity_hour = popularity_hour.merge(total_hour, how="left", on="hour").rename(columns={"aid": "candidates"})

            popularity_hour[f'popularity_hour_{c}'] = popularity_hour['count'] * 100 / popularity_hour['tot']
            popularity_hour.drop(["count", "tot"], axis=1, inplace=True)

            popularity_hour.to_parquet(f"../output/popularities/pop_h_{c}_{mode}.parquet")
            
        # Merge
        pairs = pairs.merge(popularity_week, on="candidates", how="left").fillna(0)
        pairs = pairs.merge(popularity_day, on=["candidates", "day"], how="left").fillna(0)
        pairs = pairs.merge(popularity_hour, on=["candidates", "hour"], how="left").fillna(0)
        
        pairs[[f'popularity_week_{c}', f'popularity_day_{c}', f'popularity_hour_{c}']] = pairs[[f'popularity_week_{c}', f'popularity_day_{c}', f'popularity_hour_{c}']].astype("float32")
        pairs[f'popularity_hour/day_{c}'] = (pairs[f'popularity_hour_{c}'] / pairs[f'popularity_day_{c}']).fillna(0)
        pairs[f'popularity_day/week_{c}'] = (pairs[f'popularity_day_{c}'] / pairs[f'popularity_week_{c}']).fillna(0)
        
    pairs.drop(['ts', 'day', 'hour'], axis=1, inplace=True)
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
    
    pairs[f'{feature}_rank'] = np.clip(df_ft[f'{feature}_rank'], 0, 255).astype("uint8")


def compute_matrix_factorization_features(pairs, embed, weights):
    pairs["group"] = pairs["session"] // 100000

    weights = weights.rename(columns={"candidates": "aid"})
    weightings = list(weights.columns[2:])

    fts = []
    for _, df in pairs.groupby("group"):
        df = df[["session", "candidates", "aid"]].explode("aid").reset_index(drop=True)

        df['w'] = np.sum(embed[df['candidates'].to_pandas().values] * embed[df[f'aid'].to_pandas().values], axis=1)
        df['w'] += (df['aid'] != embed.shape[0] - 1)  # non-nan are put in [0, 2]

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


def save_by_chunks(pairs, folder, part=0):
    print(f"-> Saving chunks to {folder}   (part #{part})")
    os.makedirs(folder, exist_ok=True)

    pairs["group"] = pairs["session"] // 100000

    for i, (_, df) in enumerate(pairs.groupby("group")):
        df.drop("group", axis=1, inplace=True)
        df.to_parquet(os.path.join(folder, f"{part}_{i:03d}.parquet"))

        
