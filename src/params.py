SEED = 2222
K = 5

DATA_PATH = "../input/"
OUT_PATH = "../output/"

CLASSES = ["clicks", "carts", "orders"]
WEIGHTS = [0.1, 0.3, 0.6]
NUM_CLASSES = 3

LOG_PATH = "../logs/"

NUM_WORKERS = 4

TYPE_LABELS = {"clicks": 0, "carts": 1, "orders": 2}

NEPTUNE_PROJECT = ""

GT_FILE = "../output/val_labels.parquet"
