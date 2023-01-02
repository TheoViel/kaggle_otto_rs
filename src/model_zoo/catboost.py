import sys
import lofo
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score


def objective_catboost(trial, df_train, df_val, features, target="match"):
    catboost_params = dict(
        max_depth=trial.suggest_int("max_depth", 5, 15),
        # max_leaves=trial.suggest_int("max_leaves", 100, 10000),
        gamma=trial.suggest_float("gamma", 1e-6, 1e-1, log=True),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1),
        subsample=trial.suggest_float("subsample", 0.5, 1),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 1, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 1, log=True),
    )

    model = CatBoostClassifier(
        **catboost_params,
        n_estimators=10000,
        objective="binary:logistic",
        eval_metric="auc",
        gpu_ram_part=0.95,
        gpu_cat_features_storage="GpuRam",
        task_type="GPU",
    )

    model.fit(
        df_train[features],
        df_train[target],
        eval_set=[(df_val[features], df_val[target])],
        verbose=0,
        early_stopping_rounds=20,
    )

    pred = model.predict_proba(df_val[features])[:, 1]

    y_val = (
        df_val[target].values
        if isinstance(df_val, pd.DataFrame)
        else df_val[target].get()
    )

    return roc_auc_score(y_val, pred)


def train_catboost(
    df_train,
    df_val,
    df_test,
    features,
    target="match",
    params=None,
    cat_features=[],
    i=0,
):

    model = CatBoostClassifier(
        **params,
        n_estimators=20000,
        classes_count=0,
        # boosting_type='Plain',
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="GPU",
        random_state=42 + i,
        allow_writing_files=False,
    )

    model.fit(
        df_train[features],
        df_train[target],
        eval_set=[(df_val[features], df_val[target])],
        verbose=100,
        cat_features=cat_features,
        early_stopping_rounds=100,  # None, 500
        log_cout=sys.stdout,
    )

    pred = model.predict_proba(df_val[features])[:, 1]

    return pred, model


def lofo_catboost(df, config, folds=[0], auto_group_threshold=1):
    dataset = lofo.Dataset(
        df,
        target=config.target,
        features=config.features,
        auto_group_threshold=auto_group_threshold,
    )

    cv = []
    for fold in range(config.n_folds):
        if fold in folds:
            df_train_opt = df[(df["fold_1"] != fold) & (df["fold_2"] != fold)]
            df_val_opt = df[(df["fold_1"] == fold) | (df["fold_2"] == fold)]
            cv.append((list(df_train_opt.index), list(df_val_opt.index)))

    model = CatBoostClassifier(
        **config.params,
        n_estimators=10000,
        objective="binary:logistic",
        eval_metric="auc",
        gpu_ram_part=0.95,
        gpu_cat_features_storage="GpuRam",
        task_type="GPU",
    )

    lofo_imp = lofo.LOFOImportance(dataset, scoring="roc_auc", cv=cv, model=model)

    return lofo_imp.get_importance()