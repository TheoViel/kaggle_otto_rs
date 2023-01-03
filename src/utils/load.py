import glob
import cudf
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


def load_parquets_cudf(regex, max_n=0):
    dfs = []
    for idx, chunk_file in enumerate(glob.glob(regex)):
        chunk = cudf.read_parquet(chunk_file)
        dfs.append(chunk)

        if max_n and idx >= max_n:
            break

    return cudf.concat(dfs).reset_index(drop=True)


def load_parquets_cudf(regex, max_n=0, pos_ratio=0, target=""):
    dfs = []
    for idx, chunk_file in enumerate(glob.glob(regex)):
        df = cudf.read_parquet(chunk_file)
        
        if target:
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
