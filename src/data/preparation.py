import re
import ast
import numpy as np
import pandas as pd


def clean_spaces(txt):
    txt = re.sub('\n', ' ', txt)
    txt = re.sub('\t', ' ', txt)
    txt = re.sub('\r', ' ', txt)
    return txt


def load_and_prepare(root="", lower=True):
    df = pd.read_csv(root + "codes_train.csv")

    data = pd.read_csv(root + "datasources.csv")

    data['file_name'].fillna('', inplace=True)
    data['file_extension'].fillna('', inplace=True)

    data = data.groupby("notebook_id").agg(list).reset_index()

    data['file_name'] = data.apply(
        lambda x: ", ".join([a + b for a, b in zip(x.file_name, x.file_extension)]), axis=1
    )
    data['total_rows'] = data['total_rows'].apply(np.nansum)
    data['file_size'] = data['file_size'].apply(np.nansum)

    data.drop(['total_columns', 'file_extension'], axis=1, inplace=True)
    df = df.merge(data, how="left", on="notebook_id")

    df['file_name'].fillna('', inplace=True)
    df['total_rows'].fillna(0, inplace=True)
    df['file_size'].fillna(0, inplace=True)
    df.dropna(axis=0, inplace=True)

    df['target'] = df['execution_time']
    df['is_pl'] = 0

    df['clean_text'] = df["source"].apply(clean_spaces)

    if lower:
        df['clean_text'] = df["clean_text"].apply(lambda x: x.lower())

    df = df.reset_index(drop=True)

    return df

def load_and_prepare_test(root="", lower=True):
    df = pd.read_csv(root + "codes_test.csv")

    data = pd.read_csv(root + "datasources.csv")

    data['file_name'].fillna('', inplace=True)
    data['file_extension'].fillna('', inplace=True)

    data = data.groupby("notebook_id").agg(list).reset_index()

    data['file_name'] = data.apply(
        lambda x: ", ".join([a + b for a, b in zip(x.file_name, x.file_extension)]), axis=1
    )
    data['total_rows'] = data['total_rows'].apply(np.nansum)
    data['file_size'] = data['file_size'].apply(np.nansum)

    data.drop(['total_columns', 'file_extension'], axis=1, inplace=True)
    df = df.merge(data, how="left", on="notebook_id")

    df['file_name'].fillna('', inplace=True)
    df['total_rows'].fillna(0, inplace=True)
    df['file_size'].fillna(0, inplace=True)
    df.dropna(axis=0, inplace=True)

    df['target'] = 0
    df['is_pl'] = 0

    df['clean_text'] = df["source"].apply(clean_spaces)

    if lower:
        df['clean_text'] = df["clean_text"].apply(lambda x: x.lower())

    df = df.reset_index(drop=True)

    return df
