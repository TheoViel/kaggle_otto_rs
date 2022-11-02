import gc
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

from models import NERTransformer
from training.optim import trim_tensors  # noqa
from data.loader import ClusterSampler
from data.dataset import SectionDataset
from data.tokenizer import create_tokenizer_and_tokens
from utils.metric import compute_score, get_predicted_strings
from utils.torch import seed_everything, load_model_weights

from params import SPLITS, NUM_WORKERS


def validate(
    model,
    val_dataset,
    activation="sigmoid",
    val_bs=32,
    verbose=0,
    device="cuda",
    subsample=True
):
    """
    Validation function.
    TODO : Update

    Args:
        model (torch model): Model to train.
        val_dataset (SectionDataset): Validation dataset.
        activation (str, optional): Activation function. Defaults to 'sigmoid'.
        verbose (int, optional): Period (in steps) to display logs at. Defaults to 0.
        device (str, optional): Device for torch. Defaults to "cuda".

    Returns:
        list of list of strings: Predictions.
        list of list of strings: Labels.
    """

    if subsample:
        val_sampler = ClusterSampler(
            SequentialSampler(val_dataset),
            val_dataset.clusters,
            batch_size=val_bs,
            drop_last=False,
            samples_per_cluster=1000,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            drop_last=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        print(f"- Validating on {val_sampler.n_texts} texts\n")
    else:
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            drop_last=False,
            batch_size=val_bs,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        print(f"- Validating on {len(val_dataset)} texts\n")

    model.eval()
    preds, truths, texts, probas, targets, offsets = [], [], [], [], [], []
    with torch.no_grad():
        for data in val_loader:
            ids, token_type_ids = data["ids"], data["token_type_ids"]

            # ids, token_type_ids = trim_tensors(
            #     [data["ids"], data["token_type_ids"]],
            #     model.name,
            # )

            y_pred = model(ids.cuda(), token_type_ids.cuda()).squeeze(-1)

            if activation == "sigmoid":
                y_pred = torch.sigmoid(y_pred)
            elif activation == "softmax":
                y_pred = torch.softmax(y_pred, 2)

            preds += get_predicted_strings(data, y_pred)
            truths += get_predicted_strings(data, data["target"])
            texts += data["text"]
            probas.append(y_pred.cpu().numpy())
            targets.append(data["target"].numpy().astype(int))
            offsets.append(data["offsets"].numpy().astype(int))

    return (
        preds,
        truths,
        texts,
        np.concatenate(probas),
        np.concatenate(targets),
        np.concatenate(offsets),
    )


def k_fold_val(config, df, log_folder, subsample=True):

    tokenizer, tokens = create_tokenizer_and_tokens(config)

    all_preds, all_truths, all_texts = [], [], []
    all_probas, all_targets, all_offsets = [], [], []
    for fold in range(config.k):
        if fold in config.selected_folds:
            print(f"\n-------------   Fold {fold + 1} / {config.k}  -------------")
            seed_everything(config.seed + fold)

            model = NERTransformer(
                config.name,
                nb_layers=config.nb_layers,
                nb_ft=config.nb_ft,
                k=config.conv_kernel,
                drop_p=config.drop_p,
                multi_sample_dropout=config.multi_sample_dropout,
            ).cuda()
            model.zero_grad()

            load_model_weights(
                model, f"{config.name}_{fold + 1}.pt", cp_folder=log_folder
            )

            df_val = df[df["clust"].apply(lambda x: x in SPLITS[config.k][fold])]

            val_dataset = SectionDataset(
                df_val.reset_index(drop=True),
                tokenizer,
                tokens,
                max_len=config.max_len,
                model_name=config.name,
                train=False,
            )

            preds, truths, texts, probas, targets, offsets = validate(
                model,
                val_dataset,
                verbose=config.verbose,
                device=config.device,
                subsample=subsample,
            )

            all_preds += preds
            all_truths += truths
            all_texts += texts
            all_probas.append(probas)
            all_targets.append(targets)
            all_offsets.append(offsets)

            del model, val_dataset
            torch.cuda.empty_cache()
            gc.collect()

            print(f"- Scored: {compute_score(truths, preds):.4f}\n")

    print(f"\n\nCV score : {compute_score(all_truths, all_preds):.4f}")

    return (
        all_preds,
        all_truths,
        all_texts,
        np.concatenate(all_probas),
        np.concatenate(all_targets),
        np.concatenate(all_offsets),
    )
