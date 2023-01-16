import gc
import cuml
import cudf
import glob
import numba
import pandas as pd
from sklearn.metrics import roc_auc_score
from numerize.numerize import numerize
from utils.torch import seed_everything

from model_zoo import TRAIN_FCTS, PREDICT_FCTS
from utils.load import load_parquets_cudf_folds


def train(df_train, val_regex, config, log_folder=None, optimize=False, fold=0, debug=False):
    seed_everything(config.seed)

    txt = f"{'Optimizing' if optimize else 'Training'} {config.model.upper()} Model"
    print(f"\n-------------   {txt}   -------------\n")

    if optimize:  # TODO
        raise NotImplementedError
        # study = optuna.create_study(direction="minimize")
        # objective = lambda x: objective_xgb(x, df_train, val_regex, features, target)
        # study.optimize(objective, n_trials=50)
        # print(study.best_params)
        # return study.best_params

    val_candids = sum([len(cudf.read_parquet(f, columns=['gt_orders'])) for f in glob.glob(val_regex)])
    print(f"    -> {numerize(len(df_train))} training candidates")
    print(f"    -> {numerize(val_candids)} validation candidates\n")
    
    train_fct = TRAIN_FCTS[config.model]
    df_val, model = train_fct(
        df_train,
        val_regex,
        features=config.features,
        target=config.target,
        params=config.params,
        use_es=config.use_es,
        num_boost_round=config.num_boost_round,
        folds_file=config.folds_file,
        fold=fold,
        debug=debug,
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

    # Score
    try:
        auc = roc_auc_score(df_val[config.target], df_val["pred"])
    except:
        auc = cuml.metrics.roc_auc_score(df_val[config.target].astype('int32'), df_val["pred"].values)
    
    print(f'\n -> AUC : {auc:.4f}\n')

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


def kfold(regex, test_regex, config, log_folder, debug=False):
    dfs_val, ft_imps, dfs_test = [], [], []
    for fold in range(config.k):
        print(f"\n-------------   Fold {fold + 1} / {config.k}  -------------\n")
        seed_everything(config.seed + fold)

        df_train = load_parquets_cudf_folds(
            regex,
            config.folds_file,
            fold=fold,
            pos_ratio=config.pos_ratio,
            target=config.target,
            use_gt=config.use_gt_sessions,
            train_only=True,
            columns=['session', 'candidates', 'gt_clicks', 'gt_carts', 'gt_orders'] + config.features,
            max_n=5 if debug else 0
        )

        if config.use_gt_pos:
            df_train_gt = load_parquets_cudf_folds(
                config.gt_regex,
                config.folds_file,
                fold=fold,
                pos_ratio=-1,
                target=config.target,
                use_gt=config.use_gt_sessions,
                train_only=True,
                columns=['session', 'candidates', 'gt_clicks', 'gt_carts', 'gt_orders'] + config.features,
                max_n=3 if debug else 0
            )
            df_train = pd.concat([df_train, df_train_gt], ignore_index=True)
            df_train = df_train.drop_duplicates(subset=['session', 'candidates'], keep="first").reset_index(drop=True)
            
        df_val, ft_imp, model = train(df_train, regex, config, log_folder=log_folder, fold=fold, debug=debug)
        dfs_val.append(df_val)
        ft_imps.append(ft_imp)
        
        try:
            train_sessions = set(list(df_train["session"].unique()))
            val_sessions = set(list(df_val["session"].unique().to_pandas()))
            print('Train / val sess inter', len(train_sessions.intersection(val_sessions)))
        except:
            pass
        
        if log_folder is None:
            return df_val, ft_imp, None

        predict_fct = PREDICT_FCTS[config.model]
        df_test = predict_fct(model, test_regex, config.features, debug=debug)
        dfs_test.append(df_test)
        
        print('\n -> Saving predictions \n')
        df_val[['session', 'candidates', 'pred']].to_parquet(log_folder + f"df_val_{fold}.parquet")
        
        df_test[['session', 'candidates']] = df_test[['session', 'candidates']].astype('int32')
        df_test['pred'] = df_test['pred'].astype('float32')
        df_test[['session', 'candidates', 'pred']].to_parquet(log_folder + f"df_test_{fold}.parquet")

        del df_train, df_val, ft_imp, df_test, model
        numba.cuda.current_context().deallocations.clear()
        gc.collect()

    dfs_test = cudf.concat(dfs_test).groupby(['session', 'candidates']).mean().reset_index()
    dfs_val =  cudf.concat(dfs_val).sort_values(['session', 'candidates'], ignore_index=True)
    ft_imps = pd.concat(ft_imps).reset_index().groupby('index').mean()

    if log_folder is not None:
        ft_imps.to_csv(log_folder + "ft_imp.csv")
        dfs_test[['session', 'candidates', 'pred']].to_parquet(log_folder + f"df_test.parquet")

    return dfs_val, ft_imps, dfs_test
