import sys
from params import LOG_PATH
from utils.logger import prepare_log_folder

log_folder = prepare_log_folder(LOG_PATH)
sys.exit(log_folder)
