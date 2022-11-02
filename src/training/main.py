import gc
import re
import glob
import torch
import numpy as np
import pandas as pd

from training.train import fit
from inference.predict import predict
from models import NERTransformer

from data.dataset import PatientNoteDataset
from data.tokenization import get_tokenizer
from data.preparation import load_and_prepare

from params import DATA_PATH
from utils.metric import compute_metric
from utils.torch import seed_everything, count_parameters, save_model_weights


def train(config, tokenizer, df_train, df_val, fold, df_test=None, log_folder=None):
    """
    Trains and validate a model.

    Args:
        config (Config): Parameters.
        df_train (pandas dataframe): Training metadata.
        df_val (pandas dataframe): Validation metadata.
        fold (int): Selected fold.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.

    Returns:
        np array [len(df_train) x num_classes]: Validation predictions.
        np array [len(df_train) x num_classes_aux]: Validation auxiliary predictions.
    """
    seed_everything(config.seed)

    train_dataset = PatientNoteDataset(
        df_train,
        tokenizer,
        max_len=config.max_len_train,
    )

    val_dataset = PatientNoteDataset(
        df_val,
        tokenizer,
        max_len=config.max_len,
    )
        
    if config.pretrained_weights is not None:
        if config.pretrained_weights.endswith(".pt") or config.pretrained_weights.endswith(".bin"):
            pretrained_weights = config.pretrained_weights
        else:  # folder
            pretrained_weights = glob.glob(config.pretrained_weights + f"*_{fold}.pt")[0]
    else:
        pretrained_weights = None

    model = NERTransformer(
        config.name,
        nb_layers=config.nb_layers,
        use_conv=config.use_conv,
        use_lstm=config.use_lstm,
        nb_ft=config.nb_ft,
        k=config.conv_kernel,
        drop_p=config.drop_p,
        multi_sample_dropout=config.multi_sample_dropout,
        no_dropout=config.no_dropout,
        num_classes=config.num_classes,
        pretrained_weights=pretrained_weights
    ).cuda()
    model.zero_grad()
    model.train()

    n_parameters = count_parameters(model)

    print(f"    -> {len(train_dataset)} training texts")
    print(f"    -> {len(val_dataset)} validation texts")
    print(f"    -> {n_parameters} trainable parameters\n")

    pred_val = fit(
        model,
        train_dataset,
        val_dataset,
        config.data_config,
        config.loss_config,
        config.optimizer_config,
        epochs=config.epochs,
        acc_steps=config.acc_steps,
        verbose_eval=config.verbose_eval,
        device=config.device,
        use_fp16=config.use_fp16,
        gradient_checkpointing=config.gradient_checkpointing,
    )

    if log_folder is not None:
        save_model_weights(
            model,
            f"{config.name.split('/')[-1]}_{fold}.pt",
            cp_folder=log_folder,
        )

    pred_test = 0
    if df_test is not None:
        test_dataset = PatientNoteDataset(
            df_test,
            tokenizer,
            max_len=config.max_len,
        )
        pred_test = predict(
            model, test_dataset, config.data_config, activation=config.loss_config["activation"]
        )

    del (model, train_dataset, val_dataset)
    torch.cuda.empty_cache()
    gc.collect()

    return pred_val, pred_test


def k_fold(config, df, df_extra=None, df_test=None, log_folder=None):
    tokenizer = get_tokenizer(config.name)

    folds = pd.read_csv(config.folds_file)
    df = df.merge(folds, how="left", on=["notebook_id"])

    pred_oof = np.zeros((len(df), config.num_classes))
    preds_test = []
    for fold in range(config.k):
        if fold in config.selected_folds:
            print(f"\n-------------   Fold {fold + 1} / {config.k}  -------------\n")

            seed_everything(int(re.sub(r'\W', '', config.name), base=36) % 2 ** 31 + fold)
            train_idx = list(df[df['fold'] != fold].index)
            val_idx = list(df[df['fold'] == fold].index)

            df_train = df.iloc[train_idx].copy().reset_index(drop=True)
            df_val = df.iloc[val_idx].copy().reset_index(drop=True)

            if config.extra_data_path is not None:
                if config.extra_data_path.endswith(".csv"):
                    extra_data_path = config.extra_data_path
                else:
                    extra_data_path = glob.glob(config.extra_data_path + f"*_{fold}.csv")[0]
                df_extra = pd.read_csv(extra_data_path)

                print(f'-> Using {len(df_extra)} extra samples from {extra_data_path}\n')
                df_train = pd.concat([df_train, df_extra]).reset_index(drop=True)

            pred_val, pred_test = train(
                config, tokenizer, df_train, df_val, fold, df_test=df_test, log_folder=log_folder
            )

            if log_folder is None:
                return pred_val, pred_test
            
            np.save(log_folder + f"pred_val_{fold}.npy", pred_val)
            np.save(log_folder + f"pred_test_{fold}.npy", pred_test)
            
            pred_oof[val_idx] = pred_val
            preds_test.append(pred_test)
            break

    if config.selected_folds == list(range(config.k)):
        score = compute_metric(pred_oof, df['target'])
        print(f"\n\n -> CV score : {score:.4f}")

        if log_folder is not None:
            folds.to_csv(log_folder + "folds.csv", index=False)
            np.save(log_folder + "pred_oof.npy", pred_oof)
            df.to_csv(log_folder + "df.csv", index=False)
            
    return pred_oof, np.mean(preds_test, 0)
