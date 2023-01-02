import glob
import cudf
from params import TYPE_LABELS


def load_sessions(regex):
    dfs = []
    for e, chunk_file in enumerate(glob.glob(regex)):
        chunk = cudf.read_parquet(chunk_file)
        chunk.ts = (chunk.ts / 1000).astype("int32")
        chunk["type"] = chunk["type"].map(TYPE_LABELS).astype("int8")
        dfs.append(chunk)
    
    return cudf.concat(dfs).sort_values(['session', 'aid']).reset_index(drop=True)