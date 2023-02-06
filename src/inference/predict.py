import gc
import cudf
import glob
import numba
from tqdm import tqdm


def predict_batched(
    model,
    dfs_regex,
    features,
    folds_file="",
    fold=0,
    test=False,
    debug=False,
    no_tqdm=False,
    df_val=None,
):
    """
    Batched predict function for ForestInference models.

    Args:
        model (ForestInference model): Model.
        dfs_regex (str): Regex to data.
        features (list): Features.
        folds_file (str, optional): Path to folds. Defaults to "".
        fold (int, optional): Fold. Defaults to 0.
        test (bool, optional): Whether data is test data. Defaults to False.
        debug (bool, optional): Whether to use debug mode. Defaults to False.
        no_tqdm (bool, optional): Whether to disable tqdm. Defaults to False.
        df_val (pandas DataFrame, optional): Use a preloaded dataframe instead. Defaults to None.

    Returns:
        cudf DataFrame: Results.
    """
    print("\n[Infering]")
    cols = ["session", "candidates", "gt_clicks", "gt_carts", "gt_orders", "pred"]

    if folds_file:
        folds = cudf.read_csv(folds_file)

    dfs = []
    for path in tqdm(glob.glob(dfs_regex), disable=no_tqdm):

        dfg = cudf.read_parquet(
            path, columns=features + (cols[:2] if test else cols[:5])
        )
        assert all([ft in dfg.columns for ft in features])
        if df_val is not None:
            dfg = df_val

        if folds_file:
            dfg = dfg.merge(folds, on="session", how="left")
            dfg = dfg[dfg["fold"] == fold]

        dfg = dfg.to_pandas()
        numba.cuda.current_context().deallocations.clear()
        gc.collect()

        dfg["pred"] = model.predict(dfg[features])

        numba.cuda.current_context().deallocations.clear()
        gc.collect()

        dfs.append(cudf.from_pandas(dfg[[c for c in cols if c in dfg.columns]]))

        del dfg
        numba.cuda.current_context().deallocations.clear()
        gc.collect()

        if debug or df_val is not None:
            break

    results = cudf.concat(dfs, ignore_index=True).sort_values(["session", "candidates"])
    return results
