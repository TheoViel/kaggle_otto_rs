from sklearn.preprocessing import OrdinalEncoder
import numpy as np


def encode_categories(df_train, df_test, cat_cols, nan_fill=np.nan):
    """
    This will leave all nans to nans.
    Nans must be handled later
    """
    col_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=nan_fill)

    df_train[cat_cols] = col_enc.fit_transform(df_train[cat_cols].values)

    df_test[cat_cols] = col_enc.transform(df_test[cat_cols].values)

    return df_train, df_test, col_enc


def count_encoding(df_train, df_val, df_test, feature, placeholder="id"):
    """
    Applies count encoding, we use statistics computed on train only.
    """
    counts = df_train[[feature, placeholder]].groupby(feature).count()
    mapping = counts.to_dict()[placeholder]

    encoding_train = df_train[feature].map(mapping)
    encoding_train.fillna(encoding_train.mean(), inplace=True)

    encoding_val = df_val[feature].map(mapping)
    encoding_val.fillna(encoding_train.mean(), inplace=True)

    encoding_test = df_test[feature].map(mapping)
    encoding_test.fillna(encoding_train.mean(), inplace=True)

    return encoding_train, encoding_val, encoding_test


def freq_encoding(df_train, df_val, df_test, feature, placeholder="id"):
    """
    Applies frequency encoding : number of apparition of the value.
    divided by the length of the dataframe.
    Statistics are computed separately for each dataframe.
    """
    freqs_train = df_train[[feature, placeholder]].groupby(feature).count() / len(
        df_train
    )
    freqs_val = df_val[[feature, placeholder]].groupby(feature).count() / len(df_val)

    freqs_test = df_test[[feature, placeholder]].groupby(feature).count() / len(df_test)

    mapping_train = freqs_train.to_dict()[placeholder]
    mapping_val = freqs_val.to_dict()[placeholder]
    mapping_test = freqs_test.to_dict()[placeholder]

    encoding_train = df_train[feature].map(mapping_train)
    encoding_train.fillna(encoding_train.mean(), inplace=True)

    encoding_val = df_val[feature].map(mapping_val)
    encoding_val.fillna(encoding_val.mean(), inplace=True)

    encoding_test = df_test[feature].map(mapping_test)
    encoding_test.fillna(encoding_test.mean(), inplace=True)

    return encoding_train, encoding_val, encoding_test


def target_encoding(df_train, df_val, df_test, feature, target="target"):
    """
    Applies target encoding, we use statistics computed on train only.
    """
    means = df_train[[feature, target]].groupby(feature).mean()
    mapping = means.to_dict()[target]

    encoding_train = df_train[feature].map(mapping)
    encoding_train.fillna(encoding_train.mean(), inplace=True)

    encoding_val = df_val[feature].map(mapping)
    encoding_val.fillna(encoding_train.mean(), inplace=True)

    encoding_test = df_test[feature].map(mapping)
    encoding_test.fillna(encoding_train.mean(), inplace=True)

    return encoding_train, encoding_val, encoding_test


def encode_features(
    encoding, df_train, df_val, df_test, features_to_encode, encoding_name="", log=False
):
    """
    Applies an encoding to train, val and test dataframes on given features.
    """
    for ft in features_to_encode:
        enc_train, enc_val, enc_test = encoding(df_train, df_val, df_test, ft)
        if log:
            enc_train = to_log(enc_train)
            enc_val = to_log(enc_val)
            enc_test = to_log(enc_test)
        df_train[f"{ft}_{encoding_name}_encode"] = enc_train
        df_val[f"{ft}_{encoding_name}_encode"] = enc_val
        df_test[f"{ft}_{encoding_name}_encode"] = enc_test
