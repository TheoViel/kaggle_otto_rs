import gc
import cudf
import glob
import numba
import torch
import numpy as np
from tqdm import tqdm
from merlin.loader.torch import Loader
from torch.utils.data import DataLoader

from params import NUM_WORKERS


def predict_batched(model, dfs_regex, features, folds_file="", fold=0, probs_file="", probs_mode="", ranker=False, test=False, debug=False, no_tqdm=False, df_val=None):
    print('\n[Infering]')
    cols = ['session', 'candidates', 'gt_clicks', 'gt_carts', 'gt_orders', 'pred']

    if folds_file:
        folds = cudf.read_csv(folds_file)
            
    if probs_file:
        preds = cudf.concat([
            cudf.read_parquet(f) for f in glob.glob(probs_file + "df_val_*")
        ], ignore_index=True)
        preds['pred_rank'] = preds.groupby('session').rank(ascending=False)['pred']
        assert len(preds)

    dfs = []
    for path in tqdm(glob.glob(dfs_regex), disable=no_tqdm):
        dfg = cudf.read_parquet(path, columns=features + (cols[:2] if test else cols[:5]))
        if df_val is not None:
            dfg = cudf.from_pandas(df_val)

        if folds_file:
            dfg = dfg.merge(folds, on="session", how="left")
            dfg = dfg[dfg['fold'] == fold]
    
        if probs_file:
            assert "rank" in probs_mode
            dfg = dfg.merge(preds, how="left", on=["session", "candidates"])
            max_rank = int(probs_mode.split('_')[1])
            dfg = dfg[dfg["pred_rank"] <= max_rank]
            dfg.drop(['pred', 'pred_rank'], axis=1, inplace=True)
            
#         if ranker:
#             dfg = dfg.sort_values('session', ignore_index=True)
#             group = dfg[['session', 'candidates']].groupby('session').size().to_pandas().values
#             dval = xgb.DMatrix(data=dfg[features], group=group)
#         else:

        try:
            dfg['pred'] = model.predict(dfg[features])
        except:
            dval = xgb.DMatrix(data=dfg[features])
            dfg['pred'] = model.predict(dval)
            del dval

        dfs.append(dfg[[c for c in cols if c in dfg.columns]])

        del dfg
        numba.cuda.current_context().deallocations.clear()
        gc.collect()
        
        if debug or df_val is not None:
            break

    results = cudf.concat(dfs, ignore_index=True).sort_values(['session', 'candidates'])
    return results


def predict_(model, dataset, loss_config, batch_size=64, device="cuda"):
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


def predict(model, dataset, loss_config, data_config, device="cuda"):
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
    ys = []
    cols = ['session', 'candidates', 'gt_clicks', 'gt_carts', 'gt_orders']

    loader = Loader(dataset, batch_size=data_config["val_bs"], shuffle=False)

    with torch.no_grad():
        for x, _ in tqdm(loader):
            y = torch.cat([x[k] for k in cols if k in x.keys()], 1)
            x = torch.cat([x[k] for k in data_config['features']], 1)

            # Forward
            pred = model(x)

            # Get probabilities
            if loss_config['activation'] == "sigmoid":
                pred = pred.sigmoid()
            elif loss_config['activation'] == "softmax":
                pred = pred.softmax(-1)
            preds = np.concatenate([preds, pred.cpu().numpy()])

            ys.append(y.detach().cpu().numpy())

    return preds, np.concatenate(ys)
