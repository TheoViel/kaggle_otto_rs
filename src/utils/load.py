import glob
import cudf
import pandas as pd
from tqdm import tqdm

from params import TYPE_LABELS
from utils.torch import seed_everything


def load_sessions(regexes):
    """
    Loads sessions.
    Some transformations are applied to save memory.

    Args:
        regexes (str): Regex to sessions.

    Returns:
        cudf DataFrame: Sessions
    """
    dfs = []

    if not isinstance(regexes, list):
        regexes = [regexes]

    for regex in regexes:
        for e, chunk_file in enumerate(glob.glob(regex)):
            chunk = cudf.read_parquet(chunk_file)
            chunk.ts = (chunk.ts / 1000).astype("int32")
            chunk["type"] = chunk["type"].map(TYPE_LABELS).astype("int8")
            chunk[["session", "aid"]] = chunk[["session", "aid"]].astype("int32")
            dfs.append(chunk)

    return cudf.concat(dfs).sort_values(["session", "aid"]).reset_index(drop=True)


def load_parquets_cudf_folds(
    regex,
    folds_file="",
    fold=0,
    pos_ratio=0,
    target="",
    val_only=False,
    max_n=0,
    train_only=False,
    use_gt=False,
    use_gt_for_val=False,
    columns=None,
    seed=42,
    no_tqdm=False,
):
    """
    Loads features saved in parquet files.
    Loading is done on GPU.

    Args:
        regex (str): Regex to files.
        folds_file (str, optional): Path to folds. Defaults to "".
        fold (int, optional): Fold. Defaults to 0.
        pos_ratio (int, optional): Positive ratio for downsampling negatives. Defaults to 0.
        target (str, optional): Target name. Defaults to "".
        val_only (bool, optional): Load only val data. Defaults to False.
        max_n (int, optional): Max number of chunks to load. Defaults to 0.
        train_only (bool, optional): Load only train data. Defaults to False.
        use_gt (bool, optional): Whether to filter sessions without gt. Defaults to False.
        use_gt_for_val (bool, optional): Whether to filter sessions without gt. Defaults to False.
        columns (list, optional): Columns to load. Defaults to None.
        seed (int, optional): Seed. Defaults to 42.
        no_tqdm (bool, optional): Whether to disable tqdm. Defaults to False.

    Returns:
        pandas DataFrame: Loaded data.
    """
    already_filtered = target in regex or "clicks" in target
    if already_filtered:
        assert use_gt and use_gt_for_val
        print("Files were filtered !")

    files = sorted(glob.glob(regex))
    folds = cudf.read_csv(folds_file) if folds_file else None

    if (use_gt or use_gt_for_val) and target:
        if "extra" in regex:
            GT_FILE = "../output/val_labels_trimmed.parquet"
        else:
            GT_FILE = "../output/val_labels.parquet"
        gt = cudf.read_parquet(GT_FILE)
        kept_sessions = gt[gt["type"] == target[3:]].drop("ground_truth", axis=1)

    dfs, dfs_val = [], []
    for idx, file in enumerate(tqdm(files, disable=(max_n > 0 or no_tqdm))):
        df = cudf.read_parquet(file, columns=columns)
        filtered = already_filtered

        if folds is not None:
            df = df.merge(folds, on="session", how="left")
        else:
            df["fold"] = -1

        if not filtered and (use_gt and use_gt_for_val):
            df = (
                df.merge(kept_sessions, on="session", how="left")
                .dropna(0)
                .drop("type", axis=1)
                .reset_index(drop=True)
            )
            filtered = True

        if not train_only:
            df_val = df[df["fold"] == fold].reset_index(drop=True)

            if "clicks" in target:  # Use 10% of the sessions for clicks
                df_val = df_val[df_val["session"] < df_val["session"].quantile(0.1)]

            if use_gt_for_val:
                df_val["has_gt"] = df_val.groupby("session")[target].transform("max")
                df_val = cudf.concat(
                    [
                        df_val[df_val["has_gt"] == 1],
                        df_val[df_val["has_gt"] == 0].drop_duplicates(
                            subset="session", keep="first"
                        ),
                    ],
                    ignore_index=True,
                )
                df_val.drop("has_gt", axis=1, inplace=True)

                if "carts" in target:  # Use 50% of the sessions for carts
                    df_val = df_val[df_val["session"] < df_val["session"].quantile(0.5)]

                if not filtered:
                    df_val = (
                        df_val.merge(kept_sessions, on="session", how="left")
                        .dropna(0)
                        .drop("type", axis=1)
                        .reset_index(drop=True)
                    )

            df_val = df_val.sort_values(["session", "candidates"], ignore_index=True)
            dfs_val.append(df_val.to_pandas())

        df = df[df["fold"] != fold].reset_index(drop=True)

        if val_only:
            if max_n and (idx + 1) >= max_n:
                break
            continue

        if target:  # Subsample
            if use_gt and not filtered:
                df = (
                    df.merge(kept_sessions, on="session", how="left")
                    .dropna(0)
                    .drop("type", axis=1)
                    .reset_index(drop=True)
                )

            df = df.sort_values(["session", "candidates"], ignore_index=True)
            pos = df.index[df[target] == 1]

            seed_everything(fold)

            if pos_ratio > 0:
                try:
                    n_neg = int(df[target].sum() / pos_ratio)
                    neg = (
                        df[[target]][df[target] == 0]
                        .sample(n_neg, random_state=seed)
                        .index
                    )

                    df = df.iloc[cudf.concat([pos, neg])]
                except Exception:
                    print("WARNING ! Negative sampling error, using the whole df.")

            elif pos_ratio == -1:  # only positives
                df = df.iloc[pos]
            else:
                pass
            df = df.sort_values(["session", "candidates"], ignore_index=True)
            dfs.append(df.to_pandas())
        else:
            df = df.sort_values(["session", "candidates"], ignore_index=True)
            dfs.append(df.to_pandas())
        if max_n and (idx + 1) >= max_n:
            break

    if val_only:
        return pd.concat(dfs_val, ignore_index=True)
    elif train_only:
        return pd.concat(dfs, ignore_index=True)

    return pd.concat(dfs, ignore_index=True), pd.concat(dfs_val, ignore_index=True)
