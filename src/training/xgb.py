import gc
import cuml
import cudf
import glob
import numba
import optuna
import numpy as np
import pandas as pd

from cuml import PCA
from numerize.numerize import numerize
from utils.torch import seed_everything
from sklearn.metrics import roc_auc_score

from model_zoo import TRAIN_FCTS, PREDICT_FCTS
from model_zoo.xgb import objective_xgb
from utils.load import load_parquets_cudf_folds
from utils.metrics import evaluate
from utils.plot import plot_importances


def optimize(regex, config, log_folder, n_trials=100, fold=0, debug=False, df_train=None, df_val=None, run=None):
    print(f"\n-------------  Optimizing {config.model.upper()} Model  -------------\n")
    seed_everything(config.seed)

    if df_train is None or df_val is None:
        df_train, df_val = load_parquets_cudf_folds(
            regex,
            config.folds_file,
            fold=fold,
            pos_ratio=config.pos_ratio,
            target=config.target,
            use_gt=config.use_gt_sessions,
            use_gt_for_val=True,
            columns=['session', 'candidates', 'gt_clicks', 'gt_carts', 'gt_orders'] + config.features,
            max_n=5 if debug else 0,
            probs_file=config.probs_file if config.restrict_all else "",
            probs_mode=config.probs_mode if config.restrict_all else "",
            seed=config.seed,
            no_tqdm=log_folder is not None
        )

    print(f"\n    -> {numerize(len(df_train))} training candidates")
    print(f"    -> {numerize(len(df_val))} validation candidates\n")

    study = optuna.create_study(direction="maximize")
    objective = lambda x: objective_xgb(
        x,
        df_train,
        df_val,
        regex,
        features=config.features,
        target=config.target,
        params=config.params,
        folds_file=config.folds_file,
        probs_file=config.probs_file,
        probs_mode=config.probs_mode,
        fold=0,
        debug=debug,
        no_tqdm=log_folder is not None,
        run=run,
    )

    study.optimize(objective, n_trials=1 if debug else n_trials)

    print("Final params :\n", study.best_params)
    return study


def train(df_train, df_val, regex, config, log_folder=None, fold=0, debug=False):
    print(f"\n-------------  Training {config.model.upper()} Model  -------------\n")

    print(f"    -> {numerize(len(df_train))} training candidates")
    print(f"    -> {numerize(len(df_val))} validation candidates\n")
    
    train_fct = TRAIN_FCTS[config.model]
    df_val, model = train_fct(
        df_train,
        df_val,
        regex,
        features=config.features,
        target=config.target,
        params=config.params,
        use_es=config.use_es,
        num_boost_round=config.num_boost_round,
        folds_file=config.folds_file,
        probs_file=config.probs_file,
        probs_mode=config.probs_mode,
        fold=fold,
        debug=debug,
        no_tqdm=log_folder is not None,
    )

    # Feature importance
    if config.model == "xgb":
        ft_imp = model.get_score()
    else:
        ft_imp = model.feature_importances_  # TODO
    try:
        ft_imp = pd.DataFrame(
            pd.Series(ft_imp, index=config.features), columns=["importance"]
        )
    except:
        ft_imp = None
        
    if config.mode == "test":
        return df_val, ft_imp, model

    if log_folder is None:
        return df_val, ft_imp, model

    # Save model
    if config.model == "xgb":
        model.save_model(log_folder + f"{config.model}_{fold}.json")
    elif config.model == "lgbm":
        try:
            model.booster_.save_model(log_folder + f"{config.model}_{fold}.txt")
        except Exception:
            model.save_model(log_folder + f"{config.model}_{fold}.txt")
    else:   # catboost, verif
        model.save_model(log_folder + f"{config.model}_{fold}.txt")

    return df_val, ft_imp, model


def kfold(regex, test_regex, config, log_folder, debug=False, run=None):
    seed_everything(config.seed)
    ft_imps, scores = [], []

    for fold in range(config.k):
        if fold not in config.selected_folds:
            continue

        print(f"\n=============   Fold {fold + 1} / {config.k}   =============\n")
        seed_everything(config.seed + fold)

        df_train, df_val = load_parquets_cudf_folds(
            regex,
            config.folds_file,
            fold=fold,
            pos_ratio=config.pos_ratio,
            target=config.target,
            use_gt=config.use_gt_sessions,
            use_gt_for_val=True,
            columns=['session', 'candidates', 'gt_clicks', 'gt_carts', 'gt_orders'] + config.features,
            max_n=5 if debug else 0,
            probs_file=config.probs_file if config.restrict_all else "",
            probs_mode=config.probs_mode if config.restrict_all else "",
            seed=config.seed,
            no_tqdm=log_folder is not None
        )
        
        if config.pca_components:
            print('Applying PCA')
            pca = PCA(n_components=config.pca_components, random_state=config.seed)

            pca_features = [f'pca_{i}' for i in range(config.pca_components)]
            pca.fit(
                pd.concat([df_train[config.features], df_val[config.features]], axis=0, ignore_index=True)
            )

            new_train = pca.transform(df_train[config.features])
            new_train = pd.DataFrame(new_train, columns=pca_features)
            df_train = pd.concat([df_train.drop(config.features, axis=1), new_train], axis=1)

            new_val = pca.transform(df_val[config.features])
            new_val = pd.DataFrame(new_val, columns=pca_features)
            df_val = pd.concat([df_val.drop(config.features, axis=1), new_val], axis=1)

            config.features = pca_features
        
        try:
            train_sessions = set(list(df_train["session"].unique()))
            val_sessions = set(list(df_val["session"].unique()))
            print('Train / val session inter', len(train_sessions.intersection(val_sessions)))
        except:
            pass

        if fold in config.folds_optimize:
            study = optimize(
                regex,
                config,
                log_folder,
                n_trials=1 if debug else config.n_trials,
                debug=debug,
                df_train=df_train,
                df_val=df_val,
                fold=fold,
                run=run,
            )
            config.params.update(study.best_params)
            
            if run is not None:
                run[f"fold_{fold}/best_params/"] = study.best_params

#         if config.use_gt_pos:
#             df_train_gt = load_parquets_cudf_folds(
#                 config.gt_regex,
#                 config.folds_file,
#                 fold=fold,
#                 pos_ratio=-1,
#                 target=config.target,
#                 use_gt=config.use_gt_sessions,
#                 train_only=True,
#                 columns=['session', 'candidates', 'gt_clicks', 'gt_carts', 'gt_orders'] + config.features,
#                 max_n=3 if debug else 0
#             )
#             df_train = pd.concat([df_train, df_train_gt], ignore_index=True)
#             df_train = df_train.drop_duplicates(subset=['session', 'candidates'], keep="first").reset_index(drop=True)
            
        df_val, ft_imp, model = train(
            df_train,
            df_val,
            regex,
            config,
            log_folder=log_folder,
            fold=fold,
            debug=debug
        )
        ft_imps.append(ft_imp)
        
        try:
            train_sessions = set(list(df_train["session"].unique()))
            val_sessions = set(list(df_val["session"].unique().to_pandas()))
            print('Train / val session inter', len(train_sessions.intersection(val_sessions)))
        except:
            pass
        
        if log_folder is None:
            return ft_imp
        
        if run is not None:
            score = evaluate(df_val, config.target, verbose=0)
            scores.append(score)
            run[f"fold_{fold}/recall"] = score
        
        print('\n -> Saving val predictions \n')
        df_val[['session', 'candidates', 'pred']].to_parquet(log_folder + f"df_val_{fold}.parquet")

        del df_train, df_val, ft_imp
        numba.cuda.current_context().deallocations.clear()
        gc.collect()
        
        predict_fct = PREDICT_FCTS[config.model]
        df_test = predict_fct(
            model,
            test_regex,
            config.features,
            debug=debug,
            probs_file=config.probs_file if config.restrict_all else "",
            probs_mode=config.probs_mode if config.restrict_all else "",
            ranker=("rank" in config.params["objective"]),
            no_tqdm=True,
        )
        
        print('\n -> Saving test predictions \n')
        df_test[['session', 'candidates']] = df_test[['session', 'candidates']].astype('int32')
        df_test['pred'] = df_test['pred'].astype('float32')
        df_test[['session', 'candidates', 'pred']].to_parquet(log_folder + f"df_test_{fold}.parquet")

        del df_test, model
        numba.cuda.current_context().deallocations.clear()
        gc.collect()

    ft_imps = pd.concat(ft_imps).reset_index().groupby('index').mean()
    if log_folder is not None:
        ft_imps.to_csv(log_folder + "ft_imp.csv")

    if run is not None:
        run["global/logs"].upload(log_folder + "logs.txt")
        run["global/recall"] = np.mean(scores)
        plot_importances(ft_imps, run=run)

    return ft_imps
