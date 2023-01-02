import os
from lightgbm.basic import LightGBMError
from lightgbm import LGBMClassifier, early_stopping, log_evaluation, Booster
from sklearn.metrics import roc_auc_score
from params import OUT_PATH


class CheckpointCallback:
    """
    Save model weights callback.
    """
    def __init__(self, period, **kwargs):
        self.period = period
        self.before_iteration = False
        self.kwargs = kwargs

    def __call__(self, env):
        if self.period > 0 and env.iteration >= 100 and (env.iteration + 1) % self.period == 0:
            env.model.save_model(OUT_PATH + "model_tmp.txt")


def train_lgbm(
    df_train,
    df_val,
    df_test,
    features,
    target="match",
    params=None,
    cat_features=[],
    use_es=False,
    i=0,
):
    auc = 0

    while auc < 0.98:  # hardcoded

        model = LGBMClassifier(
            **params,
            n_estimators=8000,
            objective="binary",
            device="gpu",
            random_state=42 + i,
            snapshot_freq=100,
        )

        callbacks = [log_evaluation(100), CheckpointCallback(100)]
        if use_es:
            callbacks += [early_stopping(100)]

        try:
            model.fit(
                df_train[features],
                df_train[target],
                eval_set=[(df_val[features], df_val[target])],
                eval_metric="auc",
                callbacks=callbacks,
                categorical_feature=cat_features,
            )
            pred = model.predict_proba(df_val[features])[:, 1]
            auc = 1
        except LightGBMError:
            print('LightGBMError ! Retrieving fitted model : model_tmp.txt')
            model = Booster(model_file=OUT_PATH + 'model_tmp.txt')
            pred = model.predict(df_val[features]).ravel()
            auc = roc_auc_score(df_val[target], pred)
            print(f'Fitted model scores {auc:.5f}')

        if os.path.exists(OUT_PATH + "model_tmp.txt"):
            os.remove(OUT_PATH + "model_tmp.txt")

        if auc < 0.98:
            print('\nTraining failed, retrying.\n')

    return pred, model
