import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def prepare_data(root):
    df = pd.read_csv(root + "train.csv")
    df_test = pd.read_csv(root + "test.csv")

    if "processed" not in df["path"][0]:
        df["path"] = root + "processed/" + df["path"]
    if "processed" not in df_test["path"][0]:
        df_test["path"] = root + "processed_test/" + df_test["path"]

        df["fold"] = -1
    df.loc[df.index[-2000000:], "fold"] = 0

    df_test["fold"] = -1

    return df, df_test


def json_to_pq(file, output_path="", name=None, shift_sess=False):
    """
    https://www.kaggle.com/datasets/columbia2131/otto-chunk-data-inparquet-format
    """
    if name is None:
        name = file.stem.rsplit("_", 2)[0]
    save_folder = output_path / f"{name}_parquet"
    os.makedirs(save_folder, exist_ok=True)

    print(f"Saving to {save_folder}")

    chunks = pd.read_json(file, lines=True, chunksize=100000)

    for e, chunk in enumerate(tqdm(chunks)):
        event_dict = {
            "session": [],
            "aid": [],
            "ts": [],
            "type": [],
        }

        for session, events in zip(chunk["session"].tolist(), chunk["events"].tolist()):
            for event in events:
                sess = session + int(1e8) if shift_sess else session
                event_dict["session"].append(sess)
                event_dict["aid"].append(event["aid"])
                event_dict["ts"].append(event["ts"])
                event_dict["type"].append(event["type"])

        # save DataFrame
        pd.DataFrame(event_dict).to_parquet(save_folder / f"{e:03d}.parquet")


def json_to_pq_y(file, output_path="", name=None, shift_sess=False):
    """
    https://www.kaggle.com/datasets/columbia2131/otto-chunk-data-inparquet-format
    """
    if name is None:
        name = file.stem
    df = pd.read_json(file, lines=True)

    event_dict = {
        "session": [],
        "type": [],
        "ground_truth": [],
    }

    for session, labels in df.values:
        for k, v in labels.items():
            sess = session + int(1e8) if shift_sess else session
            event_dict["session"].append(sess)
            event_dict["type"].append(k)
            event_dict["ground_truth"].append(np.array([v]).flatten())

    # save DataFrame
    pd.DataFrame(event_dict).to_parquet(output_path / f"{name}.parquet")
    print("Saved labels to ", output_path / f"{name}.parquet")

#     return pd.DataFrame(event_dict)


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
