import pandas as pd
import os



project_dir = os.path.dirname(os.path.abspath(__file__))
print(project_dir)

# List of files
files = ["P16.csv", "P19.csv", "P21.csv", "P23.csv", "P25.csv", "P26.csv", "P28.csv", 
         "P30.csv", "P34.csv", "P37.csv", "P39.csv", "P41.csv", "P42.csv", "P45.csv", 
         "P46.csv", "P56.csv", "P64.csv", "P65.csv"]

for file in files:
    #i=1
    #df_next['time'] += (i - 1) * 300
    csv_path = os.path.join(project_dir, "dataset_csv", file)
    # Load and concatenate all CSV files into a single DataFrame
    dfs = [pd.read_csv(csv_path, delimiter=';') for file in files]
    combined_df = pd.concat(dfs, ignore_index=True)
    #i=i+1

for i, df in enumerate(dfs[1:], start=2):
    df['time'] += (i - 1) * 300

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)


print(combined_df.head())

output_csv_path = os.path.join(project_dir, "dataset_csv", "test.csv")

combined_df.to_csv(output_csv_path, index=False, sep=';')

print(f"{file} processed successfully.")
