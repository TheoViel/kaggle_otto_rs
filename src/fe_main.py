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


def main(mode="val", gt=False):
    # Params
    MODE = mode
    CANDIDATES_VERSION = "cv3-tv5"
#     CANDIDATES_VERSION = "c-orders-v4"
    FEATURES_VERSION = "8"

    SUFFIX = f"{CANDIDATES_VERSION}.{FEATURES_VERSION}"
    CANDIDATE_FILE = f'../output/candidates/candidates_{CANDIDATES_VERSION}_{MODE}.parquet'
    PARQUET_FILES = f"../output/{MODE}_parquet/*"
    
    if gt:
        MODE = "val"
        CANDIDATE_FILE = f'../output/candidates/candidates_gt.parquet'
        SUFFIX = f"gt.{FEATURES_VERSION}"
        
    print(f'\n -> Generating features {SUFFIX} \n\n')

    if MODE == "val":
        OLD_PARQUET_FILES = "../output/full_train_parquet/*"
    elif MODE == "test":
        OLD_PARQUET_FILES = "../output/full_train_val_parquet/*"
    else:
        raise NotImplementedError
        
    MATRIX_FOLDER = "../output/matrices/"
    MATRIX_NAMES = [
        f"matrix_123_temporal_20_{MODE}",
        f"matrix_123_type136_20_{MODE}",
        f"matrix_12__20_{MODE}",
        f"matrix_123_type0.590.5_20_{MODE}",
        f"matrix_cpu-90_{MODE}",
        f"matrix_cpu-95_{MODE}",
        f"matrix_cpu-99_{MODE}",
        f"matrix_gpu-116_{MODE}",
        f"matrix_gpu-115_{MODE}",
        f"matrix_gpu-93_{MODE}",
        f"matrix_gpu-217_{MODE}",
#         f"matrix_gpu-220_{MODE}",
        f"matrix_gpu-226_{MODE}",
        f"matrix_gpu-232_{MODE}",
#         f"matrix_gpu-235_{MODE}",
        f"matrix_gpu-239_{MODE}",
        f"matrix_gpu-700_{MODE}",
        f"matrix_gpu-701_{MODE}",
    ]
    
    # Chunks
    CHUNK_SIZE = 1_000_000

    all_pairs = cudf.read_parquet(CANDIDATE_FILE)
    all_pairs = all_pairs.sort_values(['session', 'candidates']).reset_index(drop=True)
    
    all_pairs['group'] = all_pairs['session'] // (CHUNK_SIZE // 50)
    N_PARTS = len(all_pairs['group'].unique())
    
    all_pairs = all_pairs.to_pandas()
    numba.cuda.current_context().deallocations.clear()
    gc.collect()

    # FE loop
    for PART, (_, pairs) in enumerate(all_pairs.groupby('group')):
        print(f"\n---------   PART {PART + 1 } / {N_PARTS}   ---------\n")

        pairs.drop('group', axis=1, inplace=True)
        pairs = cudf.from_pandas(pairs).sort_values(['session', 'candidates']).reset_index(drop=True)

        # Popularity
        pairs = compute_popularity_features(pairs, [OLD_PARQUET_FILES, PARQUET_FILES], "")
        pairs = compute_popularity_features(pairs, PARQUET_FILES, "_w")

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

        # Covisitation features
        sessions = load_sessions(PARQUET_FILES)
        sessions = sessions.sort_values(['session', "aid"]).groupby('session').agg(list).reset_index()

        pairs = pairs.merge(sessions[["session", "aid"]], how="left", on="session")
        pairs = pairs.sort_values(['session', 'candidates']).reset_index(drop=True)

        for name in MATRIX_NAMES:
            print(f'-> Features from {name}')

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
        
        # Rank features
        fts_to_rank = pairs.columns[5:] if MODE == "val" else pairs.columns[2:]
        fts_to_rank = [ft for ft in fts_to_rank if not any([k in ft for k in ["_rank", "_sum", "_max"]])]
        print(f'-> Compute {len(fts_to_rank)} rank features')

        for ft in fts_to_rank:
            add_rank_feature(pairs, ft)

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

#         break

        
def parse_args():
    """
    Parses arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="val",
        help="Mode",
    )

    parser.add_argument(
        "--gt",
        default=False,
        action='store_true',
        help="Generate gt features",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    assert args.mode in ["val", "test"]

    main(args.mode, args.gt)

    print('\n\nDone !')
