import glob
import cudf
import numpy as np
import pandas as pd
from tqdm import tqdm

from params import TYPE_LABELS


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
            dfs.append(chunk)
    
    return cudf.concat(dfs).sort_values(['session', 'aid']).reset_index(drop=True)


def load_parquets_cudf_chunks_folds(regex, folds_file, fold=0, pos_ratio=0, target="", n_chunks=5, val_only=False, max_n=0, train_only=False):
    files = sorted(glob.glob(regex))
    n = int(np.ceil(len(files) / n_chunks))
    chunks = [files[c: c + n] for c in range(0, len(files), n)]
    
    folds = cudf.read_csv(folds_file)

    dfs, dfs_val = [], []
    for idx, chunk in enumerate(tqdm(chunks)):

        df = []
        for file in chunk:
            df.append(cudf.read_parquet(file))
        df = cudf.concat(df, ignore_index=True)
        
        df = df.merge(folds, on="session", how="left")
        
        df_val = df[df['fold'] == fold].reset_index(drop=True)
        df = df[df['fold'] != fold].reset_index(drop=True)
        
        if not train_only:
             dfs_val.append(df_val)
        
        if val_only:
            if max_n and (idx + 1) >= max_n:
                break
            continue
    
        if target:  # Subsample
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
        
    if val_only:
        return cudf.concat(dfs_val, ignore_index=True)
    elif train_only:
        return cudf.concat(dfs, ignore_index=True)

    return cudf.concat(dfs, ignore_index=True), cudf.concat(dfs_val, ignore_index=True)
