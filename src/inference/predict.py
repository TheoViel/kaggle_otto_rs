import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm  # noqa

from params import NUM_WORKERS
from training.optim import trim_tensors


def predict(model, dataset, data_config, activation="softmax"):
    """
    Usual predict torch function
    """
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=data_config["val_bs"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    preds = []
    with torch.no_grad():
        for data in loader:
            ids, token_type_ids = trim_tensors(
                [data["ids"], data["token_type_ids"]],
                pad_token=data_config["pad_token"],
            )

            y_pred = model(ids.cuda(), token_type_ids.cuda())

            if activation == "sigmoid":
                y_pred = y_pred.sigmoid()
            elif activation == "softmax":
                y_pred = y_pred.softmax(-1)

            preds.append(y_pred.detach().cpu().numpy())

    return np.concatenate(preds)
