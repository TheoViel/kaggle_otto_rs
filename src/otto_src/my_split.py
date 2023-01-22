import os
import random
import pandas as pd
from otto_src.testset import get_max_ts, train_test_split, create_kaggle_testset


def val_split(train_set, output_path, days=7):
    max_ts = get_max_ts(train_set)

    session_chunks = pd.read_json(train_set, lines=True, chunksize=100000)

    train_file = output_path / "train_sessions.jsonl"
    test_file_full = output_path / "val_sessions.jsonl"

    train_test_split(session_chunks, train_file, test_file_full, max_ts, days)


def train_val_split(train_set, output_path, days=7, trim=True):
    file = output_path / "sessions.jsonl"  # First 3 weeks
    val_file = output_path / "val_sessions.jsonl"  # Week 4

    max_ts = get_max_ts(train_set)
    print(f"Using {days} days before {max_ts} for val split")

    session_chunks = pd.read_json(train_set, lines=True, chunksize=100000)

    if file.exists():
        os.remove(file)
    if val_file.exists():
        os.remove(val_file)
    train_test_split(session_chunks, file, val_file, max_ts, days, trim=False)


def create_labels(file, output_path="", seed=42, last_2=False):
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
    create_kaggle_testset(sessions, sessions_file, labels_file, last_2=last_2)
