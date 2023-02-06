import cudf
import glob
import numpy as np
from multiprocessing import Pool
from params import TYPE_LABELS


def df_parallelize_run(func, t_split):
    """
    Pool multiprocessing for speedup.

    Args:
        func (function): Function to parallelize.
        t_split (list): Files.

    Returns:
        pandas DataFrame: Results.
    """
    num_cores = np.min([20, len(t_split)])
    pool = Pool(num_cores)
    df = pool.map(func, t_split)
    pool.close()
    pool.join()
    return df


def matrix_to_candids_dict(matrix):
    """
    Converts a matrix to a dict of candidates sorted by weight.

    Args:
        matrix (cudf or pandas DataFrame): Matrix.

    Returns:
        dict: {aid: [aid1, aid2, ...], ...}
    """
    matrix = matrix.sort_values(["aid_x", "wgt"], ascending=[True, False])
    candids = matrix[["aid_x", "aid_y"]].groupby("aid_x").agg(list)

    try:
        candids = candids.to_pandas()
    except AttributeError:
        pass

    candids["aid_y"] = candids["aid_y"].apply(lambda x: x.tolist())
    candids_dict = candids.to_dict()["aid_y"]

    return candids_dict


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
        chunk = cudf.read_parquet(chunk_file)

        chunk.ts = (chunk.ts / 1000).astype("int32")
        chunk["type"] = chunk["type"].map(TYPE_LABELS).astype("int8")

        chunk = chunk.sort_values(["session", "ts"])
        chunk["d"] = chunk.groupby("session").ts.diff()
        chunk["d"] = (chunk["d"] > 60 * 60 * 2).astype("int16").fillna(0)
        chunk["d"] = chunk.groupby("session").d.cumsum()

        dfs.append(chunk)

    return (
        cudf.concat(dfs).sort_values(["session", "ts"], ignore_index=True).to_pandas()
    )


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
    df = df.drop_duplicates(keep="first", subset=["session", "candidates"])

    df["candidates"] = df["candidates"].astype("uint32")
    df["session"] = df["session"].astype("uint32")

    df = df.sort_values(["session", "candidates"]).reset_index(drop=True)

    if not test:
        for col in ["gt_clicks", "gt_carts", "gt_orders"]:
            df_tgt = (
                df[["session", "candidates", col]].explode(col).reset_index(drop=True)
            ).fillna(-1)
            df_tgt[col] = df_tgt[col].astype("int64") == df_tgt["candidates"].astype(
                "int64"
            )
            assert not df_tgt.isna().any().max()

            df_tgt = df_tgt.groupby(["session", "candidates"]).max().reset_index()
            df_tgt = df_tgt.sort_values(["session", "candidates"]).reset_index(
                drop=True
            )
            assert not df_tgt.isna().any().max()

            df[col] = df_tgt[col].astype("uint8")
    return df
