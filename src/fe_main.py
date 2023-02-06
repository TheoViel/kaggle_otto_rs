import os
import gc
import re
import cudf
import numba
import warnings
import argparse
import numpy as np

from params import CLASSES
from data.fe import (
    compute_popularity_features,
    compute_popularities_new,
    compute_weights,
    load_embed,
    compute_matrix_factorization_features,
    compute_w2v_features,
    add_rank_feature,
    compute_coocurence_features,
    count_actions,
    save_by_chunks,
)
from utils.load import load_sessions

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


def main(mode="val"):
    """
    Main functions for feature engineering.

    Args:
        mode (str, optional): Mode, in ["val", "test", "extra"]. Defaults to "val".

    Raises:
        NotImplementedError: Mode not supported.
    """
    from params import GT_FILE

    # Params
    MODE = mode
    CANDIDATES_VERSION = "cv7+-tv5"

    FEATURES_VERSION = "12"

    SUFFIX = f"{CANDIDATES_VERSION}.{FEATURES_VERSION}"
    CANDIDATE_FILE = (
        f"../output/candidates/candidates_{CANDIDATES_VERSION}_{MODE}.parquet"
    )
    PARQUET_FILES = f"../output/{MODE}_parquet/*"

    if MODE == "extra":
        GT_FILE = "../output/val_labels_trimmed.parquet"  # noqa
        PARQUET_FILES = "../output/val_trimmed_parquet/*"

    print(f"\n -> Generating features {SUFFIX}  -  Mode [{MODE.upper()}]\n\n")

    if MODE == "val":
        OLD_PARQUET_FILES = "../output/full_train_parquet/*"
        CHUNK_SIZE = 50_000
    elif MODE == "test":
        OLD_PARQUET_FILES = "../output/full_train_val_parquet/*"
        CHUNK_SIZE = 50_000
    elif MODE == "extra":
        OLD_PARQUET_FILES = "../output/full_train_parquet/*"
        CHUNK_SIZE = 200_000
    else:
        raise NotImplementedError

    MATRIX_FOLDER = "../output/matrices/"

    MODE_ = "val" if MODE == "extra" else MODE
    MATRIX_NAMES = [
        f"matrix_123_temporal_20_{MODE_}",
        f"matrix_123_type136_20_{MODE_}",
        f"matrix_12__20_{MODE_}",
        f"matrix_123_type0.590.5_20_{MODE_}",
        f"matrix_cpu-90_{MODE_}",
        f"matrix_cpu-95_{MODE_}",
        f"matrix_cpu-99_{MODE_}",
        f"matrix_gpu-116_{MODE_}",
        f"matrix_gpu-115_{MODE_}",
        f"matrix_gpu-93_{MODE_}",
        f"matrix_gpu-217_{MODE_}",
        f"matrix_gpu-226_{MODE_}",
        f"matrix_gpu-232_{MODE_}",
        f"matrix_gpu-239_{MODE_}",
        f"matrix_gpu-700_{MODE_}",
        f"matrix_gpu-701_{MODE_}",
        f"matrix_gpu-155_{MODE_}",
        f"matrix_gpu-157_{MODE_}",
    ]

    EMBED_PATH = "../output/matrix_factorization/"
    EMBED_NAMES = [
        f"embed_1-9_64_cartbuy_{MODE_}.npy",
        f"embed_1_64_{MODE_}.npy",
        f"embed_1-5_64_{MODE_}.npy",
        f"embed_giba_{MODE_}.npy",
        f"word2vec_{MODE_}.emb",
    ]

    # Chunks
    all_pairs = cudf.read_parquet(CANDIDATE_FILE)

    # Remove sessions with no carts/orders for speed-up
    if MODE == "extra":  # or MODE == "val"
        gt = cudf.read_parquet(GT_FILE)
        kept_sessions = gt[gt["type"] != "clicks"].drop("ground_truth", axis=1)
        kept_sessions = kept_sessions.drop_duplicates(subset="session", keep="first")

        prev_len = len(all_pairs)
        all_pairs = (
            all_pairs.merge(kept_sessions, on="session", how="left")
            .dropna(0)
            .drop("type", axis=1)
            .reset_index(drop=True)
        )

        factor = prev_len / len(all_pairs)
        print(f"Reduced dataset size by {factor:.1f}x")
        CHUNK_SIZE = int(CHUNK_SIZE * factor)

    all_pairs = all_pairs.sort_values(["session", "candidates"]).reset_index(drop=True)
    if MODE == "extra":
        all_pairs["group"] = (
            all_pairs["session"] - all_pairs["session"].min()
        ) // CHUNK_SIZE
    else:
        all_pairs["group"] = (
            all_pairs["session"] - all_pairs["session"].min()
        ) // CHUNK_SIZE

    N_PARTS = len(all_pairs["group"].unique())

    all_pairs = all_pairs.to_pandas()
    numba.cuda.current_context().deallocations.clear()
    gc.collect()

    # FE loop
    for PART, (_, pairs) in enumerate(all_pairs.groupby("group")):
        print(f"\n---------   PART {PART + 1 } / {N_PARTS}   ---------\n")

        pairs.drop("group", axis=1, inplace=True)
        pairs = (
            cudf.from_pandas(pairs)
            .sort_values(["session", "candidates"])
            .reset_index(drop=True)
        )

        # Popularity
        pairs = compute_popularity_features(
            pairs, [OLD_PARQUET_FILES, PARQUET_FILES], ""
        )
        pairs = compute_popularity_features(pairs, PARQUET_FILES, "_w")

        sessions = load_sessions(PARQUET_FILES)
        pairs = compute_popularities_new(pairs, sessions, mode=MODE)

        del sessions
        numba.cuda.current_context().deallocations.clear()
        gc.collect()

        # Time weighting
        sessions = load_sessions(PARQUET_FILES)
        weights = compute_weights(sessions)

        pairs = pairs.merge(weights, how="left", on=["session", "candidates"])
        pairs = pairs.sort_values(["session", "candidates"]).reset_index(drop=True)

        for c in weights.columns[2:]:
            pairs[c] = pairs[c].fillna(pairs[c].min() / 2).astype("float32")

        del sessions
        numba.cuda.current_context().deallocations.clear()
        gc.collect()

        # Matrix factorization features
        for embed_name in EMBED_NAMES:
            name = embed_name.rsplit("_", 1)[0]
            print(f"-> Features from matrix {name}")

            embed = load_embed(os.path.join(EMBED_PATH, embed_name))

            # Retrieve sessions
            sessions = load_sessions(PARQUET_FILES)
            if "_cartbuy" in embed_name:
                sessions = sessions[sessions["type"] != 0]
            sessions = sessions.sort_values(["session", "ts"], ascending=[True, False])

            # Last n events
            df_s = sessions[["session", "aid"]].groupby("session").first().reset_index()
            df_s.columns = ["session", "last_0"]

            sessions["n"] = sessions[["session", "aid"]].groupby("session").cumcount()
            for n in range(5):
                if n > 0:
                    df_s = df_s.merge(
                        sessions[["session", "aid"]][sessions["n"] == n],
                        how="left",
                        on="session",
                    ).rename(columns={"aid": f"last_{n}"})
                df_s[f"last_{n}"] = (
                    df_s[f"last_{n}"].fillna(embed.shape[0] - 1).astype("int32")
                )

            pairs = pairs.merge(df_s, how="left", on="session")
            for n in range(5):
                pairs[f"last_{n}"] = (
                    pairs[f"last_{n}"].fillna(embed.shape[0] - 1).astype("int32")
                )
                pairs[f"{name}_last_{n}"] = np.sum(
                    embed[pairs["candidates"].to_pandas().values]
                    * embed[pairs[f"last_{n}"].to_pandas().values],
                    axis=1,
                )
                pairs[f"{name}_last_{n}"] -= (
                    pairs[f"last_{n}"] == embed.shape[0] - 1
                )  # nan are set to -1
                pairs.drop(f"last_{n}", axis=1, inplace=True)

            weights_noclick = None
            if "_cartbuy" in embed_name:
                sessions = load_sessions(PARQUET_FILES)
                weights_noclick = compute_weights(sessions, no_click=True)

            sessions = sessions.sort_values(["session", "ts"], ascending=[True, False])

            sessions = (
                sessions.sort_values(["session", "aid"])
                .groupby("session")
                .agg(list)
                .reset_index()
            )
            pairs = pairs.merge(sessions[["session", "aid"]], how="left", on="session")
            pairs = pairs.sort_values(["session", "candidates"]).reset_index(drop=True)

            fts = compute_matrix_factorization_features(
                pairs[["session", "candidates", "aid"]],
                embed,
                weights if weights_noclick is None else weights_noclick,
            )

            for c in fts.columns[2:]:
                pairs[f"{name}_{re.sub('w_', '', c)}"] = fts[c].values
            pairs.drop("aid", axis=1, inplace=True)

            del fts, sessions, weights_noclick, df_s
            numba.cuda.current_context().deallocations.clear()
            gc.collect()

            # W2V Benny features
            pairs = compute_w2v_features(pairs, PARQUET_FILES, embed, name)

            del embed
            numba.cuda.current_context().deallocations.clear()
            gc.collect()

        # Covisitation features
        sessions = load_sessions(PARQUET_FILES)
        sessions = (
            sessions.sort_values(["session", "aid"])
            .groupby("session")
            .agg(list)
            .reset_index()
        )

        pairs = pairs.merge(sessions[["session", "aid"]], how="left", on="session")
        pairs = pairs.sort_values(["session", "candidates"]).reset_index(drop=True)

        for name in MATRIX_NAMES:
            print(f"-> Features from {name}")

            fts = compute_coocurence_features(
                pairs[["session", "candidates", "aid"]],
                os.path.join(MATRIX_FOLDER, name + ".pqt"),
                weights,
            )

            for c in fts.columns[2:]:
                pairs[f"{name.rsplit('_', 1)[0]}_{re.sub('w_', '', c)}"] = fts[c].values

            del fts
            numba.cuda.current_context().deallocations.clear()
            gc.collect()

        pairs.drop("aid", axis=1, inplace=True)

        del sessions, weights
        numba.cuda.current_context().deallocations.clear()
        gc.collect()

        # Rank features
        fts_to_rank = pairs.columns[5:] if MODE == "val" else pairs.columns[2:]
        fts_to_rank = [
            ft
            for ft in fts_to_rank
            if not any([k in ft for k in ["_rank", "_sum", "_max"]])
        ]
        print(f"-> Compute {len(fts_to_rank)} rank features")

        for ft in fts_to_rank:
            add_rank_feature(pairs, ft)

        # Session features
        pairs = pairs.sort_values(["session", "candidates"]).reset_index(drop=True)

        for i, c in enumerate(CLASSES + ["*"]):
            print(f'-> Candidate {c if c != "*" else "views"} in session')
            sessions = load_sessions(PARQUET_FILES)

            if c != "*":
                sessions.loc[sessions["type"] != i, "aid"] = -1
            sessions = sessions.groupby("session").agg(list).reset_index()

            pairs[f"candidate_{c}_before"] = count_actions(
                pairs[["session", "candidates"]], sessions
            )

            del sessions
            numba.cuda.current_context().deallocations.clear()
            gc.collect()

        sessions = load_sessions(PARQUET_FILES)

        n_views = (
            sessions[["session", "ts"]]
            .groupby("session")
            .count()
            .reset_index()
            .rename(columns={"ts": "n_views"})
        )
        n_clicks = (
            sessions[sessions["type"] == 0][["session", "ts"]]
            .groupby("session")
            .count()
            .reset_index()
            .rename(columns={"ts": "n_clicks"})
        )
        n_carts = (
            sessions[sessions["type"] == 1][["session", "ts"]]
            .groupby("session")
            .count()
            .reset_index()
            .rename(columns={"ts": "n_carts"})
        )
        n_orders = (
            sessions[sessions["type"] == 2][["session", "ts"]]
            .groupby("session")
            .count()
            .reset_index()
            .rename(columns={"ts": "n_orders"})
        )

        sessions_fts = n_views.merge(n_clicks, how="left", on="session").fillna(0)
        sessions_fts = sessions_fts.merge(n_carts, how="left", on="session").fillna(0)
        sessions_fts = sessions_fts.merge(n_orders, how="left", on="session").fillna(0)

        for c in sessions_fts.columns[1:]:
            sessions_fts[c] = np.clip(sessions_fts[c], 0, 255).astype(np.uint8)

        pairs = pairs.merge(sessions_fts, on="session", how="left")
        pairs = pairs.sort_values(["session", "candidates"])

        # dtypes not handled by merlin dataloader
        for c in pairs.columns:
            if pairs[c].dtype == "uint16":
                pairs[c] = pairs[c].astype("int16")
            elif pairs[c].dtype == "uint32":
                pairs[c] = pairs[c].astype("int32")
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
        default="val",
        help="Mode",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    assert args.mode in ["val", "test", "extra"]
    main(args.mode)

    print("\n\nDone !")
