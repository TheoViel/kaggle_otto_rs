from model_zoo.xgb import train_xgb, objective_xgb
from model_zoo.lgbm import train_lgbm, objective_lgbm


OBJECTIVE_FCTS = {
    "xgb": objective_xgb,
    "lgbm": objective_lgbm,
}


TRAIN_FCTS = {
    "xgb": train_xgb,
    "lgbm": train_lgbm,
}
