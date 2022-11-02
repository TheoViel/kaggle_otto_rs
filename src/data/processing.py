def rescale(features, df_train, df_val, df_test=None, eps=1e-6):
    min_ = df_train[features].min()
    max_ = df_train[features].max()

    df_train[features] = (df_train[features] - min_) / (max_ - min_ + eps)
    df_val[features] = (df_val[features] - min_) / (max_ - min_ + eps)

    if df_test is not None:
        df_test[features] = (df_test[features] - min_) / (max_ - min_ + eps)

    return df_train, df_val, df_test
