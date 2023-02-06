import gc
import os
import json
import numba
import cuml
import cudf

from utils.logger import Config
from utils.metrics import evaluate
from inference.predict import predict_batched


def inference(regex, test_regex, log_folder, debug=False, save=True):
    """
    Inference function for boosting models.

    Args:
        regex (str): Regex to validation data.
        test_regex (str): Regex to test data.
        log_folder (str): Log folder.
        debug (bool, optional): Whether to use debug mode. Defaults to False.
        save (bool, optional): Whether to save predictions. Defaults to True.
    """
    config = Config(json.load(open(log_folder + "config.json", "r")))

    dfs_val = []
    for fold in range(config.k):
        if fold not in config.selected_folds:
            continue
        if fold == 0 or fold == 2:
            continue
        if f"xgb_{fold}.json" not in os.listdir(log_folder):
            continue

        print(f"\n -> Fold {fold + 1}\n")

        if config.model == "xgb":
            model = cuml.ForestInference.load(
                filename=log_folder + f"xgb_{fold}.json",
                model_type="xgboost_json",
            )
        else:
            model = cuml.ForestInference.load(
                filename=log_folder + f"lgbm_{fold}.txt",
                model_type="lightgbm",
            )

        print("- Val data")

        if regex is not None:
            df_val = predict_batched(
                model,
                regex,
                config.features,
                folds_file=config.folds_file,
                fold=fold,
                debug=debug,
            )

            evaluate(df_val, config.target)

            if not debug:
                df_val[["session", "candidates"]] = df_val[
                    ["session", "candidates"]
                ].astype("int32")
                df_val["pred"] = df_val["pred"].astype("float32")
                if save:
                    df_val[["session", "candidates", "pred"]].to_parquet(
                        log_folder + f"df_val_{fold}.parquet"
                    )

            dfs_val.append(df_val)

            if debug:
                break

        print("- Test data")

        if test_regex is not None:
            df_test = predict_batched(
                model,
                test_regex,
                config.features,
                debug=debug,
            )

            if not debug:
                df_test[["session", "candidates"]] = df_test[
                    ["session", "candidates"]
                ].astype("int32")
                df_test["pred"] = df_test["pred"].astype("float32")

                if save:
                    df_test[["session", "candidates", "pred"]].to_parquet(
                        log_folder + f"df_test_{fold}.parquet"
                    )

            del df_test
            numba.cuda.current_context().deallocations.clear()
            gc.collect()

    print("\n ===> CV :")
    evaluate(cudf.concat(dfs_val), config.target)
