from model_zoo.xgb import train_xgb, predict_batched_xgb

TRAIN_FCTS = {
    "xgb": train_xgb,
}

PREDICT_FCTS = {
    "xgb": predict_batched_xgb,
}