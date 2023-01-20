import gc
import os
import json
import numba
import xgboost

from utils.logger import Config
from model_zoo import PREDICT_FCTS
from utils.load import load_parquets_cudf_folds
from utils.metrics import evaluate


def xgb_inference(regex, test_regex, log_folder, debug=False):
    config = Config(json.load(open(log_folder + "config.json", "r")))
    
    for fold in range(config.k):
        if fold not in config.selected_folds:
            continue
        if f"xgb_{fold}.json" not in os.listdir(log_folder):
            continue

        print(f"\n -> Fold {fold + 1}\n")

        model = xgboost.Booster()
        model.load_model(log_folder + f"xgb_{fold}.json")
                
        predict_fct = PREDICT_FCTS[config.model]

        print('- Val data')
        
        df_val = predict_fct(
            model,
            regex,
            config.features,
            folds_file=config.folds_file,
            fold=fold,
            probs_file=config.probs_file if config.restrict_all else "",
            probs_mode=config.probs_mode if config.restrict_all else "",
            ranker=("rank" in config.params["objective"]),
            debug=debug,
        )

        evaluate(df_val, config.target)
        
        if not debug:
            df_val[['session', 'candidates']] = df_val[['session', 'candidates']].astype('int32')
            df_val['pred'] = df_val['pred'].astype('float32')
            df_val[['session', 'candidates', 'pred']].to_parquet(log_folder + f"df_val_{fold}.parquet")
        
        del df_val
        numba.cuda.current_context().deallocations.clear()
        gc.collect()
        
        print('- Test data')

        df_test = predict_fct(
            model,
            test_regex,
            config.features,
            probs_file=config.probs_file if config.restrict_all else "",
            probs_mode=config.probs_mode if config.restrict_all else "",
            ranker=("rank" in config.params["objective"]),
            debug=debug,
        )
       
        if not debug:
            df_test[['session', 'candidates']] = df_test[['session', 'candidates']].astype('int32')
            df_test['pred'] = df_test['pred'].astype('float32')
            df_test[['session', 'candidates', 'pred']].to_parquet(log_folder + f"df_test_{fold}.parquet")

        del df_test
        numba.cuda.current_context().deallocations.clear()
        gc.collect()
