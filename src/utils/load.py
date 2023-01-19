import gc
import os
import glob
import cudf
import numba
import numpy as np
import pandas as pd
from tqdm import tqdm

from params import TYPE_LABELS
from utils.torch import seed_everything

def load_parquets_pd(regex):
    dfs = []
    df = None
    for e, chunk_file in tqdm(enumerate(glob.glob(regex))):
        if df is None:
            df = pd.read_parquet(chunk_file)
        else:
            df.append(pd.read_parquet(chunk_file), ignore_index=True)
    return df


def load_parquets_cudf(regex, max_n=0, pos_ratio=0, target=""):
    dfs = []
    for idx, chunk_file in enumerate(glob.glob(regex)):
        df = cudf.read_parquet(chunk_file)
        
        if target:  # Negative downsampling
            df = df.reset_index(drop=True)
            pos = df.index[df[target] == 1]
            
            if pos_ratio:
                n_neg = int(df[target].sum() / pos_ratio)
                neg = df[[target]][df[target] == 0].sample(n_neg).index
                df = df.iloc[cudf.concat([pos, neg])]
            else:
                df = df.iloc[pos]

        dfs.append(df)

        if max_n and idx >= max_n:
            break

    return cudf.concat(dfs).reset_index(drop=True)


def load_parquets_cudf_chunks(regex, pos_ratio=0, target="", n_chunks=3):
    files = sorted(glob.glob(regex))
    n = int(np.ceil(len(files) / n_chunks))
    chunks = [files[c: c + n] for c in range(0, len(files), n)]

    dfs = []
    for chunk in tqdm(chunks):

        df = []
        for file in chunk:
            df.append(cudf.read_parquet(file))
        df = cudf.concat(df, ignore_index=True)
    
        if target:
            df['gt_*'] = df[['gt_carts', "gt_clicks", "gt_orders"]].max(axis=1)
            pos = df.index[df[target] == 1]

            if pos_ratio:
                n_neg = int(df[target].sum() / pos_ratio)
                neg = df[[target]][df[target] == 0].sample(n_neg).index
                df = df.iloc[cudf.concat([pos, neg])]
            else:  # only positives
                df = df.iloc[pos]

            dfs.append(df.drop('gt_*', axis=1))
            
        else:
            dfs.append(df)
            
        break

    return cudf.concat(dfs, ignore_index=True)


def load_sessions(regexes):
    dfs = []
    
    if not isinstance(regexes, list):
        regexes = [regexes]

    for regex in regexes:
        for e, chunk_file in enumerate(glob.glob(regex)):
            chunk = cudf.read_parquet(chunk_file)
            chunk.ts = (chunk.ts / 1000).astype("int32")
            chunk["type"] = chunk["type"].map(TYPE_LABELS).astype("int8")
            chunk[['session', 'aid']] = chunk[['session', 'aid']].astype("int32")
            dfs.append(chunk)
    
    return cudf.concat(dfs).sort_values(['session', 'aid']).reset_index(drop=True)

import time

def load_parquets_cudf_folds(
    regex,
    folds_file,
    fold=0,
    pos_ratio=0,
    target="",
    val_only=False,
    max_n=0,
    train_only=False,
    use_gt=False,
    use_gt_for_val=False,
    columns=None,
    probs_file="",
    probs_mode="",
    seed=42,
    no_tqdm=False,
):
    files = sorted(glob.glob(regex))    
    folds = cudf.read_csv(folds_file)
    
    if (use_gt or use_gt_for_val) and target:
        gt = cudf.read_parquet("../output/val_labels.parquet")
        kept_sessions = gt[gt['type'] == target[3:]].drop('ground_truth', axis=1)

    if probs_file:
        preds = cudf.concat([
            cudf.read_parquet(f) for f in glob.glob(probs_file + "df_val_*")
        ], ignore_index=True)
        preds['pred_rank'] = preds.groupby('session').rank(ascending=False)['pred']
        assert len(preds)

    dfs, dfs_val = [], []
    for idx, file in enumerate(tqdm(files, disable=(max_n>0 or no_tqdm))):
        df = cudf.read_parquet(file, columns=columns)
        df = df.merge(folds, on="session", how="left")
        
        
        if probs_file:
            assert "rank" in probs_mode
            df = df.merge(preds, how="left", on=["session", "candidates"])
            max_rank = int(probs_mode.split('_')[1])
            df = df[df["pred_rank"] <= max_rank]
            df.drop(['pred', 'pred_rank'], axis=1, inplace=True)

        t0 = time.time()
        if not train_only:
            df_val = df[df['fold'] == fold].reset_index(drop=True)
            
            if use_gt_for_val:
                df_val = df_val.merge(
                    kept_sessions, on="session", how="left"
                ).dropna(0).drop('type', axis=1).reset_index(drop=True)

#             t1 = time.time()
        
#             if probs_file:
#                 assert "rank" in probs_mode
#                 df_val = df_val.merge(preds, how="left", on=["session", "candidates"])
#                 max_rank = int(probs_mode.split('_')[1])
#                 df_val = df_val[df_val["pred_rank"] <= max_rank]
#                 df_val.drop(['pred', 'pred_rank'], axis=1, inplace=True)

            if "clicks" in target:  # Use 10% of the sessions for clicks
                df_val = df_val[df_val['session'] < df_val['session'].quantile(0.1)]
            
#             t2 = time.time()
            dfs_val.append(df_val.to_pandas())
#             t3 = time.time()

        df = df[df['fold'] != fold].reset_index(drop=True)

        if val_only:
            if max_n and (idx + 1) >= max_n:
                break
            continue
        
#         t4 = time.time()
        
        if target:  # Subsample
            df['gt_*'] = df[['gt_carts', "gt_clicks", "gt_orders"]].max(axis=1)
            
            if use_gt:
                df = df.merge(
                    kept_sessions, on="session", how="left"
                ).dropna(0).drop('type', axis=1).reset_index(drop=True)

            df = df.sort_values(['session', 'candidates'], ignore_index=True)
            pos = df.index[df[target] == 1]
            
#             t5 = time.time()

            seed_everything(fold)
            if pos_ratio > 0:
                try:
                    n_neg = int(df[target].sum() / pos_ratio)

#                     if probs_file:
#                         df_neg = df[["session", "candidates", target]].merge(preds, how="left", on=["session", "candidates"])
#                         df_neg = df_neg.sort_values(['session', 'candidates'], ignore_index=True)
#                         df_neg = df_neg[df_neg[target] == 0]

#                         if probs_mode == "head":
#                             neg = df_neg.sort_values('pred', ascending=False).head(n_neg).index
#                         elif "rank" in probs_mode:
#                             max_rank = int(probs_mode.split('_')[1])
#                             neg = df_neg[df_neg["pred_rank"] <= max_rank].sample(n_neg, random_state=42).index
#                         else:  # proba
#                             p = df_neg['pred'].to_pandas().values
#                             index = list(df_neg.index.to_pandas())
#                             neg = np.random.choice(index, n_neg, p=p / p.sum())
#                             neg = cudf.DataFrame(neg).set_index(0).index
#                     else:
                    neg = df[[target]][df[target] == 0].sample(n_neg, random_state=seed).index

                    df = df.iloc[cudf.concat([pos, neg])]
                except:
                    print('WARNING ! Negative sampling error, using the whole df.')
                    
                

            elif pos_ratio == -1:  # only positives
                df = df.iloc[pos]
            else:
                pass
            
#             t6 = time.time()

            dfs.append(df.drop('gt_*', axis=1).to_pandas())
        else:
            dfs.append(df.to_pandas())
            
#         t7 = time.time()

#         print(f'{t1 - t0 :.3f}')
#         print(f'{t2 - t1 :.3f}')
#         print(f'{t3 - t2 :.3f}')
#         print(f'{t4 - t3 :.3f}')
#         print(f'{t5 - t4 :.3f}')
#         print(f'{t6 - t5 :.3f}')
#         print(f'{t7 - t6 :.3f}')

        if max_n and (idx + 1) >= max_n:
            break

    if val_only:
        return pd.concat(dfs_val, ignore_index=True)
    elif train_only:
        return pd.concat(dfs, ignore_index=True)
#         return cudf.concat(dfs, ignore_index=True)

    return pd.concat(dfs, ignore_index=True), pd.concat(dfs_val, ignore_index=True)


def prepare_train_val_data(regex, folds_file, pos_ratio=0, target="", train_only=False, use_gt=False, columns=None, save_folder=""):
    files = sorted(glob.glob(regex))    
    folds = cudf.read_csv(folds_file)
    n_folds = int(folds['fold'].max()) + 1

    for idx, file in enumerate(tqdm(files)):
        df = cudf.read_parquet(file, columns=columns)
        df = df.merge(folds, on="session", how="left")
        
        for fold in range(n_folds):
            os.makedirs(save_folder + f"{fold}", exist_ok=True)
            os.makedirs(save_folder + f"{fold}/train/", exist_ok=True)
            os.makedirs(save_folder + f"{fold}/val/", exist_ok=True)

            if not train_only:
                df_val = df[df['fold'] == fold].reset_index(drop=True)

            df_train = df[df['fold'] != fold].reset_index(drop=True)

            if target:  # Subsample
                df_train['gt_*'] = df_train[['gt_carts', "gt_clicks", "gt_orders"]].max(axis=1)

                if use_gt:
                    gt = cudf.read_parquet("../output/val_labels.parquet")
                    kept_sessions = gt[gt['type'] == target[3:]].drop('ground_truth', axis=1)
                    df_train = df_train.merge(kept_sessions, on="session", how="left").dropna(0).drop('type', axis=1).reset_index(drop=True)

                pos = df_train.index[df_train[target] == 1]

                if pos_ratio > 0:
                    try:
                        n_neg = int(df_train[target].sum() / pos_ratio)
                        neg = df_train[[target]][df_train[target] == 0].sample(n_neg).index
                        df_train = df_train.iloc[cudf.concat([pos, neg])]
                    except:
                        pass
                elif pos_ratio == -1:  # only positives
                    df_train = df_train.iloc[pos]
                else:
                    pass

                df_train.drop('gt_*', axis=1, inplace=True)
                
            if not train_only:
                df_val.to_parquet(save_folder + f"{fold}/val/" + file.split('/')[-1])
                del df_val

            df_train.to_parquet(save_folder + f"{fold}/train/" + file.split('/')[-1])
            
            del df_train
            numba.cuda.current_context().deallocations.clear()
            gc.collect()
            
            break
            
        del df
        numba.cuda.current_context().deallocations.clear()
        gc.collect()
