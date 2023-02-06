import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def json_to_pq(file, output_path="", name=None, shift_sess=False):
    """
    Converts json sessions to parquet files.
    Adapted from https://www.kaggle.com/datasets/columbia2131/otto-chunk-data-inparquet-format

    Args:
        file (Path): Json file.
        output_path (str, optional): Subfolder to save to. Defaults to "".
        name (str, optional): Name of the folder to save in. Defaults to None.
        shift_sess (bool, optional): Whether to shift sessions id. Defaults to False.
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
    Converts json labels to a parquet file.

    Args:
        file (PathFile): Json file.
        output_path (str, optional): Folder to save in. Defaults to "".
        name (str, optional): Name of the file to save. Defaults to None.
        shift_sess (bool, optional): Whether to shift sessions id. Defaults to False.
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
