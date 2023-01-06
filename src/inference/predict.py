import torch
import numpy as np
from torch.utils.data import DataLoader

from params import NUM_WORKERS


def predict(model, dataset, loss_config, batch_size=64, device="cuda"):
    """
    Torch predict function.
    Args:
        model (torch model): Model to predict with.
        dataset (CustomDataset): Dataset to predict on.
        loss_config (dict): Loss config, used for activation functions.
        batch_size (int, optional): Batch size. Defaults to 64.
        device (str, optional): Device for torch. Defaults to "cuda".
    Returns:
        numpy array [len(dataset) x num_classes]: Predictions.
    """
    model.eval()
    preds = np.empty((0,  model.num_classes))
    preds_aux = []

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)

            # Forward
            pred = model(x)

            # Get probabilities
            if loss_config['activation'] == "sigmoid":
                pred = pred.sigmoid()
            elif loss_config['activation'] == "softmax":
                pred = pred.softmax(-1)
            preds = np.concatenate([preds, pred.cpu().numpy()])

    return preds
