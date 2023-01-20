import gc
import cudf
import glob
import numba
import numpy as np
import pandas as pd
import xgboost as xgb

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from utils.torch import seed_everything
from utils.metrics import evaluate
from utils.load import load_parquets_cudf, load_parquets_cudf_folds


class IterLoadForDMatrix(xgb.core.DataIter):
    def __init__(self, df=None, features=None, target=None, batch_size=256 * 1024, ranker=False):
        self.features = features
        self.target = target
        self.df = df
        self.it = 0  # set iterator to 0
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
            dt = dt.sort_values('session', ignore_index=True)
            group = dt[['session', 'candidates']].groupby('session').size().to_pandas().values
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
    probs_file="",
    probs_mode="",
    fold=0,
    debug=False,
    no_tqdm=False,
):
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
            probs_file=probs_file,
            probs_mode=probs_mode,
            columns=['session','candidates','gt_clicks','gt_carts','gt_orders'] + features,
        )
    if "rank" in params["objective"]:
        df_val = df_val.sort_values('session', ignore_index=True)
        group = df_val[['session', 'candidates']].groupby('session').size().to_pandas().values
        dval = xgb.DMatrix(data=df_val[features], label=df_val[target], group=group)
    else:
        dval = xgb.DMatrix(data=df_val[features], label=df_val[target])

    del df_train
    numba.cuda.current_context().deallocations.clear()
    gc.collect()

    # Params
    params_to_tweak = dict(
        max_depth=trial.suggest_int("max_depth", 6, 10),
        subsample=trial.suggest_float("subsample", 0.5, 1),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-5, 0.1, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-6, 1, log=True),
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
    cols = ['session', 'candidates', 'gt_clicks', 'gt_carts', 'gt_orders', 'pred']
    df_val['pred'] = model.predict(dval)
    score = evaluate(df_val[[c for c in cols if c in df_val.columns]], target)
    
    del dtrain, iter_train, dval
    numba.cuda.current_context().deallocations.clear()
    gc.collect()

    display_params = {}
    for param, v in params_to_tweak.items():
        if "sample" in param:
            display_params[param] = f'{v :.3f}'
        elif "reg_" in param:
            display_params[param] = f'{v :.2e}'
        else:
            display_params[param] = v
    print(f'Params : {display_params},\n')

    return score


def train_xgb(
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
            probs_file=probs_file,
            probs_mode=probs_mode,
            columns=['session','candidates','gt_clicks','gt_carts','gt_orders'] + features,
        )
    if "rank" in params["objective"]:
        df_val = df_val.sort_values('session', ignore_index=True)
        group = df_val[['session', 'candidates']].groupby('session').size().to_pandas().values
        dval = xgb.DMatrix(data=df_val[features], label=df_val[target], group=group)
    else:
        dval = xgb.DMatrix(data=df_val[features], label=df_val[target])

    del df_train
    numba.cuda.current_context().deallocations.clear()
    gc.collect()
    
    seed_everything(0)

    model = xgb.train(
        params,
        dtrain=dtrain,
        evals=[(dval, "val")] if use_es else None,
        num_boost_round=num_boost_round,
        early_stopping_rounds=100 if use_es else None,
        verbose_eval=100 if use_es else None,
    )

    del dtrain, iter_train
    numba.cuda.current_context().deallocations.clear()
    gc.collect()

    if no_tqdm and "clicks" in target:  # Rerun inf
        del df_val, dval
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
            ranker="rank" in params["objective"],
            debug=debug,
            no_tqdm=True,
        )
    else:
        cols = ['session', 'candidates', 'gt_clicks', 'gt_carts', 'gt_orders', 'pred']
        df_val['pred'] = model.predict(dval)
        pred_val = df_val[[c for c in cols if c in df_val.columns]]

    evaluate(pred_val, target)

    return pred_val, model


def predict_batched_xgb(model, dfs_regex, features, folds_file="", fold=0, probs_file="", probs_mode="", ranker=False, test=False, debug=False, no_tqdm=False):
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
    for path in tqdm(glob.glob(dfs_regex), disable=no_tqdm):
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
            
        if ranker:
            dfg = dfg.sort_values('session', ignore_index=True)
            group = dfg[['session', 'candidates']].groupby('session').size().to_pandas().values
            dval = xgb.DMatrix(data=dfg[features], group=group)
        else:
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
