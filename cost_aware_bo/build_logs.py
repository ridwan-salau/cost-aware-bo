import glob
import os
import pandas as pd
from json_reader import read_json

params = read_json('params')

csv_file_name = 'real_exp_logs.csv'
    
read_dir_name = 'experiment_logs'
save_dir_name = 'final_csv_logs'

read_path = f"{read_dir_name}/*.csv"
save_path = f"{save_dir_name}/{csv_file_name}"

csv_files = glob.glob(read_path)

dfs = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)

# Save the final dataframe to a new csv file
final_df.to_csv(save_path, index=False)

files = os.listdir(read_dir_name)

for file_name in files:
    file_path = os.path.join(read_dir_name, file_name)
    if os.path.isfile(file_path):
        os.remove(file_path)