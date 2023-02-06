import os
import json
import random
import pandas as pd
from tqdm import tqdm

from otto_src.testset import (
    get_max_ts,
    train_test_split,
    create_kaggle_testset,
    setEncoder,
)


def train_val_split(train_set, output_path, days=7, trim=True):
    """
    Splits the data for validation.

    Args:
        train_set (Path): Data to split.
        output_path (Path): Folder to save in.
        days (int, optional): Number of days to use for validation. Defaults to 7.
        trim (bool, optional): Whether to trim train sessions. Defaults to True.
    """
    file = output_path / "sessions.jsonl"  # First 3 weeks
    val_file = output_path / "val_sessions.jsonl"  # Week 4

    max_ts = get_max_ts(train_set)
    print(f"Using {days} days before {max_ts} for val split")

    session_chunks = pd.read_json(train_set, lines=True, chunksize=100000)

    if file.exists():
        os.remove(file)
    if val_file.exists():
        os.remove(val_file)
    train_test_split(session_chunks, file, val_file, max_ts, days, trim=trim)


def create_labels(file, output_path="", seed=42):
    """
    Creates labels by randomly splitting sessions.

    Args:
        file (Path): Sessions.
        output_path (str, optional): Path to save results to. Defaults to "".
        seed (int, optional): Seed. Defaults to 42.
    """
    sessions = pd.read_json(file, lines=True)

    name = file.stem.split("_")[0]
    name = name + "_c" if "_c" in file.stem else name

    sessions_file = output_path / f"{name}_sessions_c.jsonl"
    labels_file = output_path / f"{name}_labels.jsonl"

    if sessions_file.exists():
        os.remove(sessions_file)
    if labels_file.exists():
        os.remove(labels_file)

    random.seed(seed)
    create_kaggle_testset(sessions, sessions_file, labels_file)


def save_trimmed(
    session_chunks,
    split_file,
    max_ts,
    test_days=7,
):
    """
    Saves trimmed sessions.

    Args:
        session_chunks (pandas json): Sessions chunk.
        split_file (Path): File to save in.
        max_ts (int): Train max timestamp.
        test_days (int, optional): Number of days to use for validation. Defaults to 7.
    """
    split_millis = test_days * 24 * 60 * 60 * 1000
    split_ts = max_ts - split_millis

    split_file = open(split_file, "w")
    print(f"- Saving split sessions to {split_file}")

    for chunk in tqdm(session_chunks, desc="Splitting sessions"):
        for _, session in chunk.iterrows():
            session = session.to_dict()
            if (
                session["events"][0]["ts"] < split_ts
                and session["events"][-1]["ts"] > split_ts
            ):
                session["events"] = [
                    event for event in session["events"] if event["ts"] >= split_ts
                ]
                if len(session["events"]) > 1:
                    split_file.write(json.dumps(session, cls=setEncoder) + "\n")
    split_file.close()


def retrieve_trimmed(train_set, output_path, days=7):
    """
    Retrieves trimmed sessions.
    Trimmed sessions are sessions that start before val and end after val.
    They are truncated in the train_val_split function.

    Args:
        train_set (Path): Sessions.
        output_path (Path): Folder to save to.
        days (int, optional): Number of days to use for validation. Defaults to 7.
    """
    split_file = output_path / "val_sessions_trimmed.jsonl"  # Week 4

    max_ts = get_max_ts(train_set)
    # max_ts = 1661723999984
    print(f"Using {days} days before {max_ts} for val split")

    session_chunks = pd.read_json(train_set, lines=True, chunksize=100000)

    if split_file.exists():
        os.remove(split_file)

    save_trimmed(session_chunks, split_file, max_ts, days)
