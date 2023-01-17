import gc
import cudf
import glob
import numba
import numpy as np
import pandas as pd
import xgboost as xgb

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from utils.load import load_parquets_cudf, load_parquets_cudf_folds


def objective_xgb(trial, df_train, df_val, features, target="match"):
    xgb_params = dict(
        max_depth=trial.suggest_int("max_depth", 8, 12),
        # gamma=trial.suggest_float("gamma", 1e-6, 1e-1, log=True),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 30),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1),
        subsample=trial.suggest_float("subsample", 0.5, 1),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 1, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 1, log=True),
    )

    model = XGBClassifier(
        **xgb_params,
        n_estimators=10000,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        use_label_encoder=False,
        random_state=42,
        enable_categorical=True,
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


class IterLoadForDMatrix(xgb.core.DataIter):
    def __init__(self, df=None, features=None, target=None, batch_size=256 * 1024):
        self.features = features
        self.target = target
        self.df = df
        self.it = 0  # set iterator to 0
        self.batch_size = batch_size
        self.batches = int(np.ceil(len(df) / self.batch_size))
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
        input_data(
            data=dt[self.features], label=dt[self.target]
        )  # , weight=dt['weight'])
        self.it += 1
        return 1


def train_xgb(
    df_train,
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
    debug=False
):
    iter_train = IterLoadForDMatrix(df_train, features, target)
    dtrain = xgb.DeviceQuantileDMatrix(iter_train, max_bin=256)

    if use_es:
        if not folds_file:
            df_es = load_parquets_cudf(val_regex, max_n=5)
        else:
            df_es = load_parquets_cudf_folds(
                val_regex,
                folds_file,
                fold=fold,
                max_n=10,
                val_only=True,
                probs_file=probs_file,
                probs_mode=probs_mode,
                columns=['session','candidates','gt_clicks','gt_carts','gt_orders'] + features,
            )
        dval = xgb.DMatrix(data=df_es[features], label=df_es[target])
        del df_es

    del df_train
    numba.cuda.current_context().deallocations.clear()
    gc.collect()

    # TRAIN MODEL FOLD K
    model = xgb.train(
        params,
        dtrain=dtrain,
        evals=[(dval, "val")] if use_es else None,
        num_boost_round=num_boost_round,
        early_stopping_rounds=200 if use_es else None,
        verbose_eval=100 if use_es else None,
    )
    
    if use_es:
        del dval
    del dtrain, iter_train
    numba.cuda.current_context().deallocations.clear()
    gc.collect()

    pred_val = predict_batched_xgb(
        model,
        val_regex,
        features,
        folds_file=folds_file,
        fold=fold,
        probs_file=probs_file,
        probs_mode=probs_mode,
        debug=debug
    )

    return pred_val, model


def predict_batched_xgb(model, dfs_regex, features, folds_file="", fold=0, probs_file="", probs_mode="", test=False, debug=False):
    print('\n[Infering]')
    cols = ['session', 'candidates', 'gt_clicks', 'gt_carts', 'gt_orders', 'pred']

    if folds_file:
        folds = cudf.read_csv(folds_file)
            
    if probs_file:
        preds = cudf.concat([
            cudf.read_parquet(f) for f in glob.glob(probs_file + "df_val_*")
        ], ignore_index=True)
        preds['pred_rank'] = preds.groupby('session').rank(ascending=False)['pred']
        assert len(preds)
        
    dfs = []
    for path in tqdm(glob.glob(dfs_regex)):
        dfg = cudf.read_parquet(path, columns=features + (cols[:2] if test else cols[:5]))
        
        if folds_file:
            dfg = dfg.merge(folds, on="session", how="left")
            dfg = dfg[dfg['fold'] == fold]
    
        if probs_file:
            assert "rank" in probs_mode
            dfg = dfg.merge(preds, how="left", on=["session", "candidates"])
            max_rank = int(probs_mode.split('_')[1])
            dfg = dfg[dfg["pred_rank"] <= max_rank]
            dfg.drop(['pred', 'pred_rank'], axis=1, inplace=True)

        dval = xgb.DMatrix(data=dfg[features])

        dfg['pred'] = model.predict(dval)
        dfs.append(dfg[[c for c in cols if c in dfg.columns]])

        del dval, dfg
        numba.cuda.current_context().deallocations.clear()
        gc.collect()
        
        if debug:
            break

    results = cudf.concat(dfs, ignore_index=True).sort_values(['session', 'candidates'])
    return results
