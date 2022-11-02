cd src


log_folder=$(python get_log_folder.py 2>&1)
printf "Logging results to $log_folder \n\n"

model="tf_efficientnetv2_s"
epochs=15

python main_2.py --fold 0 --log_folder ${log_folder} --model ${model} --epochs ${epochs} & python main_2.py --fold 1 --log_folder ${log_folder} --model ${model} --epochs ${epochs} & python main_2.py --fold 2 --log_folder ${log_folder} --model ${model} --epochs ${epochs} & python main_2.py --fold 3 --log_folder ${log_folder} --model ${model} --epochs ${epochs}


# log_folder=$(python get_log_folder.py 2>&1)
# printf "Logging results to $log_folder \n\n"

# model="tf_efficientnetv2_s"
# epochs=15

# python main.py --fold 0 --log_folder ${log_folder} --model ${model} --epochs ${epochs} & python main.py --fold 1 --log_folder ${log_folder} --model ${model} --epochs ${epochs} & python main.py --fold 2 --log_folder ${log_folder} --model ${model} --epochs ${epochs} & python main.py --fold 3 --log_folder ${log_folder} --model ${model} --epochs ${epochs}

