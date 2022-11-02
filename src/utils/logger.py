import os
import re
import sys
import json
import shutil
import datetime
import subprocess
import numpy as np


class Config:
    """
    Placeholder to load a config from a saved json
    """
    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)


class Logger(object):
    """
    Simple logger that saves what is printed in a file
    """
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def create_logger(directory="", name="logs.txt"):
    """
    Creates a logger to log output in a chosen file
    Args:
        directory (str, optional): Path to save logs at. Defaults to "".
        name (str, optional): Name of the file to save the logs in. Defaults to "logs.txt".
    """
    log = open(directory + name, "a", encoding="utf-8")
    file_logger = Logger(sys.stdout, log)

    sys.stdout = file_logger
    sys.stderr = file_logger


def prepare_log_folder(log_path):
    """
    Creates the directory for logging.
    Logs will be saved at log_path/date_of_day/exp_id
    Args:
        log_path (str): Directory
    Returns:
        str: Path to the created log folder
    """
    today = str(datetime.date.today())
    log_today = f"{log_path}{today}/"

    if not os.path.exists(log_today):
        os.mkdir(log_today)

    exps = []
    for f in os.listdir(log_today):
        try:
            exps.append(int(f))
        except:
            continue
    exp_id = np.max(exps) + 1 if len(exps) else 0

    log_folder = log_today + f"{exp_id}/"

    assert not os.path.exists(log_folder), "Experiment already exists"
    os.mkdir(log_folder)

    return log_folder


def save_config(config, path):
    """
    Saves a config as a json and pandas dataframe.
    Args:
        config (Config): Config.
        path (str): Path to save at.
    Returns:
        pandas dataframe: Config as a dataframe.
    """
    dic = config.__dict__.copy()
    del (dic["__doc__"], dic["__module__"], dic["__dict__"], dic["__weakref__"])

    if not path.endswith('.json'):
        path += ".json"

    with open(path, "w") as f:
        json.dump(dic, f)
