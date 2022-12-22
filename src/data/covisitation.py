import os
import gc
import cudf
import numba
import numpy as np

from tqdm import tqdm


def read_file(f, data_cache):
    return cudf.DataFrame(data_cache[f])


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

    numba.cuda.current_context().deallocations.clear()
#     return matrix