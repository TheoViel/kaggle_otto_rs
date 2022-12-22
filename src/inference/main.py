import glob
import json
import numpy as np
import pandas as pd

from models import NERTransformer
from inference.predict import predict

from data.dataset import PatientNoteDataset
from data.tokenization import get_tokenizer
from data.processing import preds_to_labels
from data.post_processing import post_process_spaces
from data.preparation import load_and_prepare_test, load_and_prepare

from utils.logger import Config
from utils.metric import micro_f1
from utils.torch import load_model_weights


def padded_mean(arrays, weights=None):
    weights = np.ones(len(arrays)) if weights is None else weights

    for i in range(len(arrays)):
        if len(arrays[i].shape) == 1:
            arrays[i] = arrays[i][:, None]

    max_len = np.max([x.shape[0] for x in arrays])
    arrays = [
        np.concatenate([x, np.zeros((max_len - x.shape[0], x.shape[1]))])
        for x in arrays
    ]
    return np.average(arrays, axis=0, weights=weights)


def blend(preds, weights=None):
    weights = np.ones(len(preds)) if weights is None else weights
    if not all([pred[0].shape[1] > 1 for pred in preds]):
        for i, pred in enumerate(preds):
            if pred[0].shape[1] > 1:
                preds[i] = [p[:, 1:].sum(-1, keepdims=True) for p in pred]
    return [
        padded_mean([preds[i][j] for i in range(len(preds))], weights)
        for j in range(len(preds[0]))
    ]


def to_array(preds, max_len=1000):
    array = np.zeros((len(preds), max_len), dtype=preds[0].dtype)
    for i in range(len(preds)):
        length = min(max_len, len(preds[i].flatten()))
        array[i, :length] = preds[i].flatten()[:length]

    return array


def inference_val(df, exp_folder, save=False, cfg_folder=None):
    config = Config(json.load(open(exp_folder + "config.json", "r")))

    folds = pd.read_csv(exp_folder + "folds.csv")
    df = df.merge(folds, how="left", on=["case_num", "pn_num"])

    cfg_folder = (
        None if cfg_folder is None else cfg_folder + config.name.split("/")[-1] + "/"
    )
    model_config_file = None if cfg_folder is None else cfg_folder + "config.pth"
    tokenizer_folder = None if cfg_folder is None else cfg_folder + "tokenizers/"

    tokenizer = get_tokenizer(
        config.name, precompute=config.precompute_tokens, df=df, folder=tokenizer_folder
    )

    try:
        _ = config.use_lstm
    except AttributeError:
        config.use_lstm = False

    model = NERTransformer(
        config.name,
        nb_layers=config.nb_layers,
        use_lstm=config.use_lstm,
        use_conv=config.use_conv,
        nb_ft=config.nb_ft,
        k=config.conv_kernel,
        drop_p=config.drop_p,
        multi_sample_dropout=config.multi_sample_dropout,
        num_classes=config.num_classes,
        config_file=model_config_file,
        pretrained=False,
    ).cuda()
    model.zero_grad()

    weights = sorted(glob.glob(exp_folder + "*.pt"))
    assert len(weights) == config.k, "Missing weights"

    pred_oof = [[] for i in range(len(df))]
    for fold, weight in enumerate(weights):
        assert weight.endswith(
            f"_{fold}.pt"
        ), f"Weights name {weight} does not match fold {fold}"

        val_idx = list(df[df["fold"] == fold].index)
        df_val = df.iloc[val_idx].copy().reset_index(drop=True)

        dataset = PatientNoteDataset(
            df_val,
            tokenizer,
            max_len=config.max_len,
        )

        model = load_model_weights(model, weight)

        pred_val = predict(
            model,
            dataset,
            data_config=config.data_config,
            activation=config.loss_config["activation"],
        )
        for i, val_i in enumerate(val_idx):
            pred_oof[val_i] = pred_val[i]

    df["preds"] = preds_to_labels(pred_oof)
    df["preds_pp"] = df.apply(
        lambda x: post_process_spaces(x["preds"], x["clean_text"]), 1
    )

    score = micro_f1(df["preds_pp"], df["target"])
    print(f"\n\n -> CV score : {score:.4f}")

    if save:
        np.save(exp_folder + "pred_oof.npy", np.array(pred_oof, dtype=object))
        df.to_csv(exp_folder + "df.csv", index=False)

    return pred_oof


def inference_test(
    exp_folders, data_folder="", cfg_folder=None, debug=False, excluded_weights=None
):
    preds = []
    for exp_id, exp_folder in enumerate(exp_folders):
        config = Config(json.load(open(exp_folder + "config.json", "r")))
        config.max_len = 512

        if debug:
            df = load_and_prepare(root=data_folder, lower=config.lower)
        else:
            df = load_and_prepare_test(root=data_folder, lower=config.lower)

        if cfg_folder is not None:
            model_config_file = cfg_folder + config.name.split("/")[-1] + "/config.pth"
            tokenizer_folder = cfg_folder + config.name.split("/")[-1] + "/tokenizers/"
        else:
            model_config_file, tokenizer_folder = None, None

        tokenizer = get_tokenizer(
            config.name,
            precompute=config.precompute_tokens,
            df=df,
            folder=tokenizer_folder,
        )

        dataset = PatientNoteDataset(
            df,
            tokenizer,
            max_len=config.max_len,
        )

        try:
            _ = config.use_lstm
        except AttributeError:
            config.use_lstm = False

        model = NERTransformer(
            config.name,
            nb_layers=config.nb_layers,
            use_conv=config.use_conv,
            use_lstm=config.use_lstm,
            nb_ft=config.nb_ft,
            k=config.conv_kernel,
            drop_p=config.drop_p,
            multi_sample_dropout=config.multi_sample_dropout,
            num_classes=config.num_classes,
            config_file=model_config_file,
            pretrained=False,
        ).cuda()
        model.zero_grad()

        weights = sorted(glob.glob(exp_folder + "*.pt"))
        for fold, weight in enumerate(weights):

            if excluded_weights is not None:
                if fold in excluded_weights[exp_id]:
                    continue

            model = load_model_weights(model, weight)

            pred = predict(
                model,
                dataset,
                data_config=config.data_config,
                activation=config.loss_config["activation"],
            )
            preds.append(pred)

            if debug and len(preds) > 1:
                break

    return preds


def inference_pl(df, exp_folders, fold=0, cfg_folder=None):
    preds = []
    for exp_folder in exp_folders:
        config = Config(json.load(open(exp_folder + "config.json", "r")))

        if config.lower:
            df["feature_text"] = df["ft_ref"].apply(lambda x: x.lower())
            df["clean_text"] = df["text_ref"].apply(lambda x: x.lower())
        else:
            df["feature_text"] = df["ft_ref"]
            df["clean_text"] = df["text_ref"]

        if cfg_folder is not None:
            model_config_file = cfg_folder + config.name.split("/")[-1] + "/config.pth"
            tokenizer_folder = cfg_folder + config.name.split("/")[-1] + "/tokenizers/"
        else:
            model_config_file, tokenizer_folder = None, None

        tokenizer = get_tokenizer(
            config.name,
            precompute=config.precompute_tokens,
            df=df,
            folder=tokenizer_folder,
        )

        dataset = PatientNoteDataset(
            df,
            tokenizer,
            max_len=config.max_len,
        )

        model = NERTransformer(
            config.name,
            nb_layers=config.nb_layers,
            use_conv=config.use_conv,
            nb_ft=config.nb_ft,
            k=config.conv_kernel,
            drop_p=config.drop_p,
            multi_sample_dropout=config.multi_sample_dropout,
            num_classes=config.num_classes,
            config_file=model_config_file,
            pretrained=False,
        ).cuda()
        model.zero_grad()

        weights = sorted(glob.glob(exp_folder + f"*_{fold}.pt"))
        for weight in weights:
            model = load_model_weights(model, weight)

            pred = predict(
                model,
                dataset,
                data_config=config.data_config,
                activation=config.loss_config["activation"],
            )
            preds.append(pred)

    return preds
