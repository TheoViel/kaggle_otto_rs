import cuml
import cudf
import numpy as np
from merlin.io import Dataset
from numerize.numerize import numerize
from sklearn.metrics import roc_auc_score

from training.train import fit
from inference.predict import predict
from model_zoo.mlp import define_model
from utils.torch import seed_everything, count_parameters


def train(train_files, val_files, config, log_folder=None):
    seed_everything(config.seed)

    print(f"\n-------------   Training {config.model.upper()} Model   -------------\n")
    
    train_dataset = Dataset(train_files)

    if config.mode != "test":
        val_dataset = Dataset(val_files[:1])
    else:
        val_dataset = None
    
    model = define_model(
        name=config.model,
        nb_ft=config.nb_ft,
        d=config.d,
        p=config.p,
        num_layers=config.num_layers,
        num_classes=config.num_classes
    ).cuda()
    model.zero_grad()
        
        
    n_val = sum([len(cudf.read_parquet(f, columns=['gt_orders'])) for f in val_files[:1]])
    n_train  = sum([len(cudf.read_parquet(f, columns=['gt_orders'])) for f in train_files])
    print(f"    -> {numerize(n_train)} training candidates")
    print(f"    -> {numerize(n_val)} validation candidates subset")
    print(f"    -> {numerize(count_parameters(model))} trainable parameters\n")
    
    pred_val = fit(
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

    val_candids = sum([len(cudf.read_parquet(f, columns=['gt_orders'])) for f in val_files])
    print(f"\n    -> Inferring {numerize(val_candids)} candidates\n")

    dataset =  Dataset(val_files)
    preds, y = predict(model, dataset, config.loss_config, config.data_config)
    
    cols = ['session', 'candidates']
    if y.shape[-1] == 5:
        cols += ['gt_clicks', 'gt_carts', 'gt_orders']
    if preds.shape[1] == 3:
        cols += ['pred_clicks', 'pred_carts', 'pred_orders']
    else:
        cols += ['pred']
        
    results = cudf.DataFrame(np.concatenate([y, preds], 1), columns=cols)
    results = results.sort_values(['session', 'candidates']).reset_index(drop=True)
    
    if config.mode == "test":
        return results

    # Score
    print()
    for i, tgt in enumerate(config.target):
        auc = cuml.metrics.roc_auc_score(results[tgt].astype('int32'), results["pred" + tgt[2:]].values)
        print(f'-> {tgt} - AUC : {auc:.4f}')
    print()

    if log_folder is None:
        return results

    # Save stuff
    to_save = ['session', 'candidates']
    for c in ['pred_clicks', 'pred_carts', 'pred_orders']:
        if c in results.columns:
            results[c] = results[c].astype("float32")
            to_save.append(c)
    results[to_save].to_parquet(log_folder + "results.parquet")

    return results
