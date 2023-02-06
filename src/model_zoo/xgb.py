import gc
import cudf
import numba
import numpy as np
import xgboost as xgb

from utils.torch import seed_everything
from utils.metrics import evaluate
from utils.load import load_parquets_cudf_folds


class IterLoadForDMatrix(xgb.core.DataIter):
    """
    Loader for optimized memory use by Chris Deotte.
    """
    def __init__(
        self, df=None, features=None, target=None, batch_size=256 * 1024, ranker=False
    ):
        self.features = features
        self.target = target
        self.df = df
        self.it = 0
        self.batch_size = batch_size
        self.batches = int(np.ceil(len(df) / self.batch_size))
        self.ranker = ranker
        super().__init__()

    def reset(self):
        """Reset the iterator"""
        self.it = 0

    def next(self, input_data):
        """Yield next batch of data."""
        if self.it == self.batches:
            return 0  # Return 0 when there's no more batch.

        a = self.it * self.batch_size
        b = min((self.it + 1) * self.batch_size, len(self.df))
        dt = cudf.DataFrame(self.df.iloc[a:b])

        if self.ranker:
            dt = dt.sort_values("session", ignore_index=True)
            group = (
                dt[["session", "candidates"]]
                .groupby("session")
                .size()
                .to_pandas()
                .values
            )
            input_data(data=dt[self.features], label=dt[self.target], group=group)
        else:
            input_data(data=dt[self.features], label=dt[self.target])
        self.it += 1
        return 1


def objective_xgb(
    trial,
    df_train,
    df_val,
    val_regex,
    features=[],
    target="",
    params=None,
    num_boost_round=10000,
    folds_file="",
    fold=0,
    debug=False,
    no_tqdm=False,
    run=None,
):
    """
    Optimizes an XGBoost Classifier.

    Args:
        trial (optuna trial): Optuna trial.
        df_train (cudf or pandas DataFrame): Train data.
        df_val (cudf or pandas DataFrame): Val data.
        val_regex (str): Regex to val data. Only used if df_val is None.
        features (list, optional): Features. Defaults to [].
        target (str, optional): _description_. Defaults to "".
        params (dict, optional): Boosting parameters. Defaults to None.
        num_boost_round (int, optional): Number of boosting rounds. Defaults to 10000.
        folds_file (str, optional): Path to folds. Defaults to "".
        fold (int, optional): Fold. Defaults to 0.
        debug (bool, optional): Whether to use debug mode. Defaults to False.
        no_tqdm (bool, optional): Whether to disable tqdm. Defaults to False.
        run (Neptune run, optional): Run for logging. Defaults to None.

    Returns:
        float: Recall@20
    """
    # Data
    seed_everything(0)

    iter_train = IterLoadForDMatrix(
        df_train, features, target, ranker="rank" in params["objective"]
    )
    dtrain = xgb.DeviceQuantileDMatrix(iter_train, max_bin=256)

    if df_val is None:
        df_val = load_parquets_cudf_folds(
            val_regex,
            folds_file,
            fold=fold,
            max_n=1 if debug else 10,
            val_only=True,
            columns=["session", "candidates", "gt_clicks", "gt_carts", "gt_orders"]
            + features,
        )

    dval = xgb.DMatrix(data=df_val[features], label=df_val[target])

    del df_train
    numba.cuda.current_context().deallocations.clear()
    gc.collect()

    # Params
    params_to_tweak = dict(
        max_depth=trial.suggest_int("max_depth", 5, 10),
        subsample=trial.suggest_float("subsample", 0.4, 1),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1),
        reg_alpha=trial.suggest_float("reg_alpha", 0.01, 100, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 0.01, 100, log=True),
    )
    params.update(params_to_tweak)

    # Model
    seed_everything(0)

    model = xgb.train(
        params,
        dtrain=dtrain,
        evals=[(dval, "val")],
        num_boost_round=num_boost_round,
        early_stopping_rounds=100,
        verbose_eval=100,
    )

    # Eval & verbose
    cols = ["session", "candidates", "gt_clicks", "gt_carts", "gt_orders", "pred"]
    df_val["pred"] = model.predict(dval)
    score = evaluate(df_val[[c for c in cols if c in df_val.columns]], target)

    if run is not None:
        run[f"fold_{fold}/recall_opt"].log(score)

    del dtrain, iter_train, dval
    numba.cuda.current_context().deallocations.clear()
    gc.collect()

    display_params = {}
    for param, v in params_to_tweak.items():
        if "sample" in param:
            display_params[param] = f"{v :.3f}"
        elif "reg_" in param:
            display_params[param] = f"{v :.2e}"
        else:
            display_params[param] = v
    print(f"Params : {display_params},\n")

    return score


def train_xgb(
    df_train,
    df_val,
    val_regex,
    features=[],
    target="",
    params=None,
    num_boost_round=10000,
    folds_file="",
    fold=0,
    debug=False,
    no_tqdm=False,
):
    """
    Trains an XGBoost Classifier.

    Args:
        df_train (cudf or pandas DataFrame): Train data.
        df_val (cudf or pandas DataFrame): Val data.
        val_regex (str): Regex to val data. Only used if df_val is None.
        features (list, optional): Features. Defaults to [].
        target (str, optional): _description_. Defaults to "".
        params (dict, optional): Boosting parameters. Defaults to None.
        num_boost_round (int, optional): Number of boosting rounds. Defaults to 10000.
        folds_file (str, optional): Path to folds. Defaults to "".
        fold (int, optional): Fold. Defaults to 0.
        debug (bool, optional): Whether to use debug mode. Defaults to False.
        no_tqdm (bool, optional): Whether to disable tqdm. Defaults to False.

    Returns:
        cudf DataFrame: Results.
        xgb model : Model.
    """
    seed_everything(0)

    iter_train = IterLoadForDMatrix(
        df_train, features, target, ranker="rank" in params["objective"]
    )
    dtrain = xgb.DeviceQuantileDMatrix(iter_train, max_bin=256)

    if df_val is None:
        df_val = load_parquets_cudf_folds(
            val_regex,
            folds_file,
            fold=fold,
            max_n=1 if debug else 10,
            val_only=True,
            columns=["session", "candidates", "gt_clicks", "gt_carts", "gt_orders"]
            + features,
        )

    dval = xgb.DMatrix(data=df_val[features], label=df_val[target])

    del df_train
    numba.cuda.current_context().deallocations.clear()
    gc.collect()

    seed_everything(0)

    model = xgb.train(
        params,
        dtrain=dtrain,
        evals=[(dval, "val")],
        num_boost_round=num_boost_round,
        early_stopping_rounds=100,
        verbose_eval=100,
    )

    del dtrain, iter_train
    numba.cuda.current_context().deallocations.clear()
    gc.collect()

    cols = ["session", "candidates", "gt_clicks", "gt_carts", "gt_orders", "pred"]
    df_val["pred"] = model.predict(dval)
    pred_val = df_val[[c for c in cols if c in df_val.columns]]

    evaluate(pred_val, target)

    return pred_val, model
