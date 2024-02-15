import pandas as pd
import os

# Determine project directory
project_dir = os.path.dirname(os.path.abspath(__file__))
print(project_dir)

# List of files
files = ["P16.csv", "P19.csv", "P21.csv", "P23.csv", "P25.csv", "P26.csv", "P28.csv", 
         "P30.csv", "P34.csv", "P37.csv", "P39.csv", "P41.csv", "P42.csv", "P45.csv", 
         "P46.csv", "P56.csv", "P64.csv", "P65.csv"]

for file in files:
    file_eda = file.replace(".csv", ".csv_peaks.csv")

    # Load the CSV file using the project directory
    csv_path_valence = os.path.join(project_dir, "../datasets/arousal_classes_csv", file)
    csv_path_eda = os.path.join(project_dir, "../datasets/ECG_peaks_determined_csv", file_eda)

    # Load the EDA peaks DataFrame
    df = pd.read_csv(csv_path_eda, delimiter=';')
    df_valence = pd.read_csv(csv_path_valence, delimiter=';')

    # Resample the DataFrame to keep every 40th row
    df = df.iloc[::40]
    df.reset_index(drop=True, inplace=True)

    # Merge the 'classes' column from df_valence to df based on index
    df['classes'] = df_valence['classes']
    
    # df['classes'].fillna(0, inplace=True)
    # df['classes'] = df['classes'].astype(int)
    # Convert 'classes' column to integer type


    # Save the resampled DataFrame to a CSV file
    output_csv_path = os.path.join(project_dir, "../datasets/dataset_csv_ecg_arousal", f"{file}")

    df.to_csv(output_csv_path, index=False, sep=';')

    print(f"{file} processed successfully.")
