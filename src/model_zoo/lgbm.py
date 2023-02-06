# NOT USED BUT KEPT FOR LEGACY


import warnings
import lightgbm as lgb

from utils.torch import seed_everything
from utils.metrics import evaluate

warnings.simplefilter(action="ignore", category=UserWarning)


def objective_lgbm(
    trial,
    df_train,
    df_val,
    val_regex,
    features=[],
    target="",
    params=None,
    num_boost_round=10000,
    folds_file="",
    probs_file="",
    probs_mode="",
    fold=0,
    debug=False,
    no_tqdm=False,
    run=None,
):
    # Params
    params_to_tweak = dict(
        num_leaves=trial.suggest_int("num_leaves", 16, 512),
        subsample=trial.suggest_float("subsample", 0.4, 1),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1),
        reg_alpha=trial.suggest_float("reg_alpha", 0.01, 100, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 0.01, 100, log=True),
    )
    params.update(params_to_tweak)

    # Model
    seed_everything(0)

    model = lgb.LGBMRanker(
        **params, n_estimators=num_boost_round, random_state=10, metric="auc", n_jobs=20
    )

    group_train = df_train[['session', 'candidates']].groupby('session').size().values
    group_val = df_val[['session', 'candidates']].groupby('session').size().values

    model.fit(
        df_train[features],
        df_train[target],
        group=group_train,
        verbose=100,
        early_stopping_rounds=100,
        eval_set=[(df_val[features], df_val[target])],
        eval_group=[group_val],
        eval_at=[20],
    )

    # Eval & verbose
    cols = ["session", "candidates", "gt_clicks", "gt_carts", "gt_orders", "pred"]
    df_val["pred"] = model.predict(df_val[features])
    score = evaluate(df_val[[c for c in cols if c in df_val.columns]], target)

    if run is not None:
        run[f"fold_{fold}/recall_opt"].log(score)

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


def train_lgbm(
    df_train,
    df_val,
    val_regex,
    features=[],
    target="",
    params=None,
    cat_features=[],  # TODO
    use_es=0,
    num_boost_round=10000,
    folds_file="",
    probs_file="",
    probs_mode="",
    fold=0,
    debug=False,
    no_tqdm=False,
):
    seed_everything(0)

    model = lgb.LGBMClassifier(
        **params, n_estimators=num_boost_round, random_state=10, metric="auc", n_jobs=20
    )

    group_train = df_train[["session", "candidates"]].groupby("session").size().values
    group_val = df_val[["session", "candidates"]].groupby("session").size().values

    model.fit(
        df_train[features],
        df_train[target],
        group=group_train,
        verbose=100,
        early_stopping_rounds=100,
        eval_set=[(df_val[features], df_val[target])],
        eval_group=[group_val],
        eval_at=[20],
    )

    cols = ["session", "candidates", "gt_clicks", "gt_carts", "gt_orders", "pred"]
    df_val["pred"] = model.predict(df_val[features])
    pred_val = df_val[[c for c in cols if c in df_val.columns]]

    evaluate(pred_val, target)

    return pred_val, model
