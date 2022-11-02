import os
import glob
import torch
import warnings
import argparse
import numpy as np
import pandas as pd

from params import DATA_PATH
from data.preparation import load_and_prepare, load_and_prepare_test
from utils.logger import create_logger, save_config


def parse_args():
    """
    Parses arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold number",
    )
    parser.add_argument(
        "--log_folder",
        type=str,
        default="",
        help="Folder to log results to",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model name",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0,
        help="Learning rate",
    )
    parser.add_argument(
        "--gpu_offset",
        type=int,
        default=0,
        help="Offset for gpu selection",
    )

    return parser.parse_args()


BATCH_SIZES = {
    "microsoft/deberta-v3-base": 32,
    "microsoft/codebert-base": 32,
    "microsoft/deberta-v3-large": 32,
}

LRS = {
    "microsoft/deberta-v3-base": 3e-5,
    "microsoft/codebert-base": 3e-5,
    "microsoft/deberta-v3-large": 3e-5,
}

MAX_LENS = {
    "microsoft/deberta-v3-base": 512,
    "microsoft/codebert-base": 512,
    "microsoft/deberta-v3-large": 512,
}

class Config:
    # General
    seed = 2222
    device = "cuda"
    
    # Splits
    k = 4
    random_state = 2222
    selected_folds = [0, 1, 2, 3]
    folds_file = "/workspace/folds_kgd_4.csv"

    # Architecture
    name = "microsoft/codebert-base"

    pretrained_weights = None 

    no_dropout = False
    use_conv = False
    use_lstm = False
    nb_layers = 1
    nb_ft = 128
    conv_kernel = 5
    drop_p = 0 if no_dropout else 0.1
    multi_sample_dropout = False
    num_classes = 1

    # Texts
    max_len_train = MAX_LENS[name]
    max_len = 512
    lower = True

#     extra_data_path = OUT_PATH + "pl_case5/"
    extra_data_path = None  # OUT_PATH + "pl_6/df_pl.csv"

    # Training    
    loss_config = {
        "name": "mse",  # dice, ce, bce
        "smoothing": 0,  # 0.01
        "activation": "",  # "sigmoid", "softmax"
    }

    data_config = {
        "batch_size": BATCH_SIZES[name],
        "val_bs": BATCH_SIZES[name] * 2,
        "use_len_sampler": True,
        "pad_token": 1 if "roberta" in name else 0,
    }

    optimizer_config = {
        "name": "AdamW",
        "lr": 5e-5,
        "lr_transfo": LRS[name],
        "lr_decay": 0.99,
        "warmup_prop": 0.1,
        "weight_decay": 1,
        "betas": (0.5, 0.99),
        "max_grad_norm": 1.,
        # AWP
        "use_awp": False,
        "awp_start_step": 1000,
        "awp_lr": 1,
        "awp_eps": 5e-5 if "xlarge" in name else 1e-3,
        "awp_period": 3,
        # SWA
        "use_swa": False,
        "swa_start": 9400,
        "swa_freq": 500,
    }

    gradient_checkpointing = False
    acc_steps = 1
    epochs = 1

    use_fp16 = True

    verbose = 1
    verbose_eval = 1000

    
if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)

    print('Starting')
    args = parse_args()
    
#     print(torch.cuda.device_count())
    fold = args.fold
    print("Using GPU ", fold)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(fold + args.gpu_offset)
    assert torch.cuda.device_count() == 1

    log_folder = args.log_folder
    create_logger(directory=log_folder, name=f"logs_{fold}.txt")
    
    print("Device :", torch.cuda.get_device_name(0), "\n")
    
    # Data
    df = load_and_prepare(DATA_PATH)

    df_test = load_and_prepare_test(DATA_PATH)

    config = Config
    config.selected_folds = [fold]
    
    if args.model:
        config.name = args.model
        
    if args.epochs:
        config.epochs = args.epochs

    if args.lr:
        config.optimizer_config["lr_transfo"] = args.epochs

    print(f'- Model  {config.name}')
    print(f'- Epochs {config.epochs}')
    print(f'- LR {config.optimizer_config["lr_transfo"]}')

    save_config(config, log_folder + "config.json")
    
    print('\n -> Training\n')

    from training.main import k_fold
    k_fold(config, df, df_test=df_test, log_folder=log_folder)

    print('\nDone !')
