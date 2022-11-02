cd src

log_folder="/tmp/logs/2022-10-24/0/"
printf "Running inference for $log_folder \n\n"

python main_inf.py --fold 0 --log_folder ${log_folder} & python main_inf.py --fold 1 --log_folder ${log_folder} & python main_inf.py --fold 2 --log_folder ${log_folder} & python main_inf.py --fold 3 --log_folder ${log_folder}
