import gc
import cuml
import cudf
import numba
import numpy as np
import pandas as pd

from merlin.io import Dataset
from numerize.numerize import numerize
from sklearn.metrics import roc_auc_score

from training.train import fit
from inference.predict import predict
from model_zoo.mlp import define_model
from utils.torch import seed_everything, count_parameters
from utils.metrics import get_coverage
from utils.load import load_sessions
from params import CLASSES, WEIGHTS


def train(train_files, val_files, test_files, config, fold=0, log_folder=None, debug=False):
    seed_everything(config.seed)
    
    if debug:
        train_files = train_files[:10]
        config.epochs = 1
        val_files = val_files[:10]
        test_files = test_files[:10]

    print(f"\n-------------   Training {config.model.upper()} Model   -------------\n")
    
    train_dataset = Dataset(train_files)
    val_dataset = Dataset(val_files[:10])
    
    model = define_model(
        name=config.model,
        nb_ft=config.nb_ft,
        d=config.d,
        p=config.p,
        num_layers=config.num_layers,
        num_classes=config.num_classes
    ).cuda()
    model.zero_grad()

    n_val = sum([len(cudf.read_parquet(f, columns=['gt_orders'])) for f in val_files[:10]])
    n_train  = sum([len(cudf.read_parquet(f, columns=['gt_orders'])) for f in train_files])
    print(f"    -> {numerize(n_train)} training candidates")
    print(f"    -> {numerize(n_val)} validation candidates subset")
    print(f"    -> {numerize(count_parameters(model))} trainable parameters\n")

    fit(
        model,
        train_dataset,
        val_dataset,
        config.data_config,
        config.loss_config,
        config.optimizer_config,
        epochs=config.epochs,
        verbose_eval=config.verbose_eval,
        use_fp16=config.use_fp16,
        run=None,
    )
    
    del train_dataset, val_dataset
    numba.cuda.current_context().deallocations.clear()
    gc.collect()

    val_candids = sum([len(cudf.read_parquet(f, columns=['gt_orders'])) for f in val_files])
    print(f"\n    -> Inferring {numerize(val_candids)} val candidates\n")

    dataset =  Dataset(val_files)
    preds, y = predict(model, dataset, config.loss_config, config.data_config)

    cols = ['session', 'candidates', 'gt_clicks', 'gt_carts', 'gt_orders', 'pred_clicks', 'pred_carts', 'pred_orders']
    df_val = cudf.DataFrame(np.concatenate([y, preds], 1), columns=cols)
    df_val[cols[:2]] = df_val[cols[:2]].astype("int32")
    df_val[cols[2:5]] = df_val[cols[2:5]].astype("uint8")
    df_val[cols[5:]] = df_val[cols[5:]].astype("float32")
    df_val = df_val.sort_values(['session', 'candidates']).reset_index(drop=True)

    # Score
    print()
    for i, tgt in enumerate(config.target):
        auc = cuml.metrics.roc_auc_score(df_val[tgt].astype('int32'), df_val["pred" + tgt[2:]].values)
        print(f'-> {tgt} - AUC : {auc:.4f}')
    print()
    
    evaluate(df_val)

    if log_folder is None:
        return df_val

    df_val[cols].to_parquet(log_folder + f"df_val_{fold}.parquet")
    
    del dataset, preds, y, df_val
    numba.cuda.current_context().deallocations.clear()
    gc.collect()
    
    # Test inference
    test_candids = sum([len(cudf.read_parquet(f, columns=['session'])) for f in test_files])
    print(f"\n    -> Inferring {numerize(test_candids)} test candidates per chunk\n")
    
    n = int(np.ceil(len(test_files) / 10))  # in parts
    chunks = [test_files[c: c + n] for c in range(0, len(test_files), n)]
    
    for c, chunk in enumerate(chunks):
        dataset =  Dataset(chunk)
        preds, y = predict(model, dataset, config.loss_config, config.data_config)

        cols = ['session', 'candidates', 'pred_clicks', 'pred_carts', 'pred_orders']
        df_test = cudf.DataFrame(np.concatenate([y, preds], 1).astype(np.float32), columns=cols)
        df_test[cols[:2]] = df_test[cols[:2]].astype("int32")
        df_test[cols[2:]] = df_test[cols[2:]].astype("float32")
        df_test[cols].to_parquet(log_folder + f"df_test_{fold}_{c}.parquet")

        del dataset, preds, y, df_test
        numba.cuda.current_context().deallocations.clear()
        gc.collect()
        

def evaluate(df_val):
    dfs = load_sessions(f"../output/val_parquet/*")
    gt = pd.read_parquet("../output/val_labels.parquet")

    # Post-process
    preds = df_val[['session']].drop_duplicates(keep="first").sort_values('session', ignore_index=True).to_pandas()

    for idx, c in enumerate(CLASSES):
        preds_c = df_val.sort_values(['session', f'pred_{c}'], ascending=[True, False])
        preds_c = preds_c[['session', 'candidates', f'pred_{c}']].groupby('session').agg(list).reset_index()

        preds_c = preds_c.to_pandas()
        preds_c['candidates'] = preds_c['candidates'].apply(lambda x: x[:20])

        # Fill less than 20 candidates. This should be useless in the future
        top = dfs.loc[dfs["type"] == idx, "aid"].value_counts().index.values[:20].tolist()
        preds_c['candidates'] = preds_c['candidates'].apply(lambda x: list(x) + top[:20 - len(x)])

        preds_c = preds_c.sort_values('session')
        preds[f"candidates_{c}"] = preds_c["candidates"].values
        preds[f'pred_{c}'] = preds_c[f'pred_{c}'].values
        
    del dfs, preds_c
    numba.cuda.current_context().deallocations.clear()
    gc.collect()

    recalls = []
    for col in CLASSES:
        if f"gt_{col}" not in preds.columns:
            preds = preds.merge(gt[gt["type"] == col].drop("type", axis=1), how="left").rename(
                columns={"ground_truth": f"gt_{col}"}
            )

        n_preds, n_gts, n_found = get_coverage(
            preds[f"candidates_{col}"].values, preds[f"gt_{col}"].values
        )

        print(
            f"- {col} \t-  Found {numerize(n_found)} GTs\t-  Recall : {n_found / n_gts :.4f}"
        )
        recalls.append(n_found / n_gts)

    cv = np.average(recalls, weights=WEIGHTS)
    print(f"\n-> CV : {cv:.4f}")
