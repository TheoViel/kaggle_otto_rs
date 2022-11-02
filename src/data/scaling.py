import numpy as np


def scale(df_train, df_val, df_test, feature):
    """
    Scales specified feature of split dataframes between 0 and 1.
    We use statistics computed on train only
    :param df_train: Train dataframe
    :param df_val: Validation dataframe
    :param df_test: Test dataframe
    :param feature: feature to scale
    :return: Scaled features for train, val and test
    """
    max_ = np.max(df_train[feature])
    min_ = np.min(df_train[feature])

    return (
        (df_train[feature] - max_) / (max_ - min_),
        (df_val[feature] - max_) / (max_ - min_),
        (df_test[feature] - max_) / (max_ - min_),
    )


def to_log(feature):
    """
    Log scales a feature
    :param feature: Series to scale
    :return: Log scaled feature
    """
    assert feature.isna().sum() == 0
    assert np.min(feature) >= 0

    return np.log(1 + feature.astype(float))


def auto_log_scale(df_train, df_val, df_test, features, categorical_features=[]):
    """
    Looks for features that have a too high amplitude and applies log scaling to them
    We use statistics computed on train only
    :param df_train: Train dataframe
    :param df_val: Validation dataframe
    :param df_test: Test dataframe
    :param features: Features to check
    :param categorical_features: Categorical features to exclude from the research
    :return: Features that were scaled.
    """
    log_features = []
    for feature in features:
        if (
            df_train[feature].max() / (df_train[feature].min() + 1) > 100
            and feature not in categorical_features
        ):
            if df_train[feature].min() >= 0 and df_val[feature].min() >= 0:
                df_train[feature] = to_log(df_train[feature])
                df_val[feature] = to_log(df_val[feature])
                df_test[feature] = to_log(df_test[feature])
                log_features.append(feature)
    return log_features
