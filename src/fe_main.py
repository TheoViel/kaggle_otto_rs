import os
import gc
import re
import cudf
import glob
import numba
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from params import *
from data.fe import *
from utils.load import load_sessions


def main(mode="train"):
    # Params
    MODE = mode
    CANDIDATES_VERSION = "v3"
    FEATURES_VERSION = "5"

    SUFFIX = f"{CANDIDATES_VERSION}.{FEATURES_VERSION}"
    CANDIDATE_FILE = f'../output/candidates/candidates_{CANDIDATES_VERSION}_{MODE}.parquet'
    PARQUET_FILES = f"../output/{MODE}_parquet/*"

    if MODE == "val":
        OLD_PARQUET_FILES = "../output/full_train_parquet/*"
    elif MODE == "train":
        OLD_PARQUET_FILES = "../output/other_parquet/*"
    elif MODE == "test":
        OLD_PARQUET_FILES = "../output/full_train_val_parquet/*"
    else:
        raise NotImplementedError
        
    MATRIX_FOLDER = "../output/matrices/"
    MATRIX_NAMES = [f"matrix_123_temporal_20_{MODE}", f"matrix_123_type136_20_{MODE}", f"matrix_12__20_{MODE}", f"matrix_123_type0.590.5_20_{MODE}"]

    # Chunks
    pairs = cudf.read_parquet(CANDIDATE_FILE)
    pairs = pairs.sort_values(['session', 'candidates']).reset_index(drop=True)
    
    CHUNK_SIZE = 25_000_000
    N_PARTS = int(np.ceil(len(pairs) / CHUNK_SIZE))
    
    # FE loop
    for PART in range(N_PARTS):
        print(f"\n---------   PART {PART + 1 } / {N_PARTS}   ---------\n")

        # Subsample
        pairs = cudf.read_parquet(CANDIDATE_FILE)
        pairs = pairs.sort_values(['session', 'candidates']).reset_index(drop=True)
        
        ids = np.arange(PART * CHUNK_SIZE, min((PART + 1) * CHUNK_SIZE, len(pairs)))
        pairs = pairs.iloc[ids].reset_index(drop=True)


        # Time weighting
        sessions = load_sessions(PARQUET_FILES)
        weights = compute_weights(sessions)

        pairs = pairs.merge(weights, how="left", on=["session", "candidates"])
        pairs = pairs.sort_values(['session', 'candidates']).reset_index(drop=True)

        for c in weights.columns[2:]:
            pairs[c] = pairs[c].fillna(pairs[c].min() / 2).astype("float32")

        del sessions
        numba.cuda.current_context().deallocations.clear()
        gc.collect()

        # Popularity
        pairs = compute_popularity_features(pairs, [OLD_PARQUET_FILES, PARQUET_FILES], "")
        pairs = compute_popularity_features(pairs, OLD_PARQUET_FILES, "_old")
        pairs = compute_popularity_features(pairs, PARQUET_FILES, "_w")

        # Covisitation features
        sessions = load_sessions(PARQUET_FILES)
        sessions = sessions.sort_values(['session', "aid"]).groupby('session').agg(list).reset_index()

        pairs = pairs.merge(sessions[["session", "aid"]], how="left", on="session")
        pairs = pairs.sort_values(['session', 'candidates']).reset_index(drop=True)

        for name in MATRIX_NAMES:
            print(f' -> Features from {name}')

            fts = compute_coocurence_features(
                pairs[['session', 'candidates', 'aid']],
                os.path.join(MATRIX_FOLDER, name + ".pqt"),
                weights
            )

            for c in fts.columns[2:]:
                pairs[f"{name.rsplit('_', 1)[0]}_{re.sub('w_', '', c)}"] = fts[c].values

            del fts
            numba.cuda.current_context().deallocations.clear()
            gc.collect()

        pairs.drop('aid', axis=1, inplace=True)

        del sessions, weights
        numba.cuda.current_context().deallocations.clear()
        gc.collect()

        # Session features
        pairs = pairs.sort_values(['session', 'candidates']).reset_index(drop=True)

        for i, c in enumerate(CLASSES + ["*"]):
            print(f'-> Candidate {c if c != "*" else "views"} in session')
            sessions = load_sessions(PARQUET_FILES)

            if c != "*":
                sessions.loc[sessions["type"] != i, "aid"] = -1
            sessions = sessions.groupby('session').agg(list).reset_index()

            pairs[f'candidate_{c}_before'] = count_actions(
                pairs[['session', 'candidates']],
                sessions
            )

            del sessions
            numba.cuda.current_context().deallocations.clear()
            gc.collect()

        sessions = load_sessions(PARQUET_FILES)

        n_views = sessions[['session', 'ts']].groupby('session').count().reset_index().rename(columns={"ts": "n_views"})
        n_clicks = sessions[sessions['type'] == 0][['session', 'ts']].groupby('session').count().reset_index().rename(columns={"ts": "n_clicks"})
        n_carts = sessions[sessions['type'] == 1][['session', 'ts']].groupby('session').count().reset_index().rename(columns={"ts": "n_carts"})
        n_orders = sessions[sessions['type'] == 2][['session', 'ts']].groupby('session').count().reset_index().rename(columns={"ts": "n_orders"})

        sessions_fts = n_views.merge(n_clicks, how="left", on="session").fillna(0)
        sessions_fts = sessions_fts.merge(n_carts, how="left", on="session").fillna(0)
        sessions_fts = sessions_fts.merge(n_orders, how="left", on="session").fillna(0)

        for c in sessions_fts.columns[1:]:
            sessions_fts[c] = np.clip(sessions_fts[c], 0, 255).astype(np.uint8)

        pairs = pairs.merge(sessions_fts, on="session", how="left")
        pairs = pairs.sort_values(['session', 'candidates'])
        
        # dtypes not handled by merlin dataloader
        for c in pairs.columns:
            if pairs[c].dtype == "uint16":
                pairs[c] = pairs[c].astype('int16')
            elif pairs[c].dtype == "uint32":
                pairs[c] = pairs[c].astype('int32')
            else:
                continue

        save_by_chunks(pairs, f"../output/features/fts_{MODE}_{SUFFIX}/", part=PART)
        
        del pairs, sessions_fts, sessions, n_clicks, n_views, n_carts, n_orders
        numba.cuda.current_context().deallocations.clear()
        gc.collect()

        
        
def parse_args():
    """
    Parses arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="Mode",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    assert args.mode in ["train", "val", "test"]

    main(args.mode)
    
    print('\n\nDone !')
