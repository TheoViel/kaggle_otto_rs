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


def json_to_pq(file, output_path=""):
    """
    https://www.kaggle.com/datasets/columbia2131/otto-chunk-data-inparquet-format
    """
    name = file.stem.split("_")[0]
    save_folder = output_path / f"{name}_parquet"
    os.makedirs(save_folder, exist_ok=True)

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
                event_dict["session"].append(session)
                event_dict["aid"].append(event["aid"])
                event_dict["ts"].append(event["ts"])
                event_dict["type"].append(event["type"])

        # save DataFrame
        pd.DataFrame(event_dict).to_parquet(save_folder / f"{e:03d}.parquet")


def json_to_pq_y(file, output_path=""):
    """
    https://www.kaggle.com/datasets/columbia2131/otto-chunk-data-inparquet-format
    """
    name = file.stem
    df = pd.read_json(file, lines=True)

    event_dict = {
        "session": [],
        "type": [],
        "ground_truth": [],
    }

    for session, labels in df.values:
        for k, v in labels.items():
            event_dict["session"].append(session)
            event_dict["type"].append(k)
            event_dict["ground_truth"].append(np.array([v]).flatten())

    # save DataFrame
    pd.DataFrame(event_dict).to_parquet(output_path / f"{name}.parquet")
    print("Saved labels to ", output_path / f"{name}.parquet")

#     return pd.DataFrame(event_dict)
