import pandas as pd
import os

project_dir = os.path.dirname(os.path.abspath(__file__))
print(project_dir)

# List of files
files = ["P16.csv", "P19.csv", "P21.csv", "P23.csv", "P25.csv", "P26.csv", "P28.csv",
         "P30.csv", "P34.csv", "P37.csv", "P39.csv", "P41.csv", "P42.csv", "P45.csv",
         "P46.csv", "P56.csv", "P64.csv", "P65.csv"]

# Initialize an empty list to store DataFrames
dfs = []

# Load and concatenate all CSV files into a single DataFrame
for file in files:
    csv_path = os.path.join(project_dir, "dataset_csv_ecg_arousal", file) # dataset_csv_eda_valence, dataset_csv_eda_arousal
    df = pd.read_csv(csv_path, delimiter=';')
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

# Adjust the 'time' column for consecutive files
for i, df in enumerate(dfs[1:], start=1):
    df['time'] += i * 300

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

print(combined_df.head())

output_csv_path = os.path.join(project_dir, "dataset_csv_ecg_arousal", "merged_dataset.csv") #dataset_csv_eda_valence dataset_csv_eda_arousal

combined_df.to_csv(output_csv_path, index=False, sep=';')

print("All files processed successfully.")

