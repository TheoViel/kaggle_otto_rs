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
    probs_file="",
    probs_mode="",
    ranker=False,
    test=False,
    debug=False,
    no_tqdm=False,
    df_val=None,
):
    print("\n[Infering]")
    cols = ["session", "candidates", "gt_clicks", "gt_carts", "gt_orders", "pred"]

    if folds_file:
        folds = cudf.read_csv(folds_file)

    if probs_file:
        preds = cudf.concat(
            [cudf.read_parquet(f) for f in glob.glob(probs_file + "df_val_*")],
            ignore_index=True,
        )
        preds["pred_rank"] = preds.groupby("session").rank(ascending=False)["pred"]
        assert len(preds)

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

        if probs_file:
            assert "rank" in probs_mode
            dfg = dfg.merge(preds, how="left", on=["session", "candidates"])
            max_rank = int(probs_mode.split("_")[1])
            dfg = dfg[dfg["pred_rank"] <= max_rank]
            dfg.drop(["pred", "pred_rank"], axis=1, inplace=True)

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
