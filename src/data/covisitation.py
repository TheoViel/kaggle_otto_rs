import os
import gc
import cudf
import numba
import pandas as pd

from tqdm import tqdm
from params import TYPE_LABELS


def read_file_to_cache(f):
    """
    Reads a parquet file to cache.

    Args:
        f (str): File to load.

    Returns:
        pandas DataFrame: Loaded file.
    """
    df = pd.read_parquet(f)
    df.ts = (df.ts / 1000).astype("int32")
    df["type"] = df["type"].map(TYPE_LABELS).astype("int8")
    return df


def read_file(f, data_cache):
    """
    Pandas Dataframe -> cudf Dataframe.

    Args:
        f (str): File key.
        data_cache (dict): File cache.

    Returns:
        cudf DataFrame: File.
    """
    return cudf.DataFrame(data_cache[f])


def compute_covisitation_matrix(
    files,
    data_cache,
    weighting="",
    type_weight={},
    considered_types=[1, 2, 3],
    n=0,
    save_folder="",
    suffix="",
):
    """
    Computes Theo's covisitation matrices.

    Args:
        files (list): Filenames.
        data_cache (dict): Data cache.
        weighting (str, optional): Weighting type, in "temporal", "type", "". Defaults to "".
        type_weight (dict, optional): Type weights. Defaults to {}.
        considered_types (list, optional): Considered types. Defaults to [1, 2, 3].
        n (int, optional): Number of aids to consider. Defaults to 0.
        save_folder (str, optional): Folder to save the matrix in. Defaults to "".
        suffix (str, optional): Save suffix ("test" or "val"). Defaults to "".
    """
    DISK_PIECES = 4
    SIZE = 1.86e6 / DISK_PIECES

    chunks = [files[x: x + 10] for x in range(0, len(files), 10)]

    matrices = []
    for part in range(DISK_PIECES):
        for idx, chunk in enumerate(tqdm(chunks)):
            df = cudf.concat(
                [read_file(file, data_cache) for file in chunk], ignore_index=True
            )

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

            # SAVE MEM
            df = df.loc[(df.aid_x >= part * SIZE) & (df.aid_x < (part + 1) * SIZE)]

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
            if idx == 0:
                matrix = df
            else:
                matrix = matrix.add(df, fill_value=0)

            del df
            numba.cuda.current_context().deallocations.clear()
            gc.collect()

        # CONVERT MATRIX TO DICTIONARY
        matrix = matrix.reset_index()
        matrix = matrix.sort_values(["aid_x", "wgt"], ascending=[True, False])

        # SAVE TOP N
        matrix = matrix.reset_index(drop=True)
        matrix["n"] = matrix.groupby("aid_x").aid_y.cumcount()

        if n:
            matrix = matrix.loc[matrix.n < n].drop("n", axis=1)

        matrices.append(matrix.to_pandas())

    if save_folder:
        if weighting == "type":
            weighting += "".join(map(str, list(type_weight.values())))
        save_path = os.path.join(
            save_folder,
            f'matrix_{"".join(map(str, considered_types))}_{weighting}_{n}_{suffix}.pqt',
        )
        print(f"Saving matrix to {save_path}")
        pd.concat(matrices, ignore_index=True).to_parquet(save_path)

    numba.cuda.current_context().deallocations.clear()
