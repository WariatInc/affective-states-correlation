import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os

# # Determine project directory
# project_dir = os.path.dirname(os.path.abspath(__file__))
# print(project_dir)
# file="P16.csv"
# file_eda=file.replace(".csv", ".csv_peaks.csv")


# # Load the CSV file using the project directory

# csv_path_valence = os.path.join(project_dir, "valance_classes_csv",  file)
# csv_path_eda = os.path.join(project_dir, "EDA_peaks_determined_csv", file_eda)


# # # Load and merge the datasets
# df_valence = pd.read_csv(csv_path_valence)
# df_eda = pd.read_csv(csv_path_eda)

# print(df_eda.head())
# df_eda['time'] = pd.to_datetime(df_eda['time'], format='%S')

# # Set 'time' column as the index
# df_eda.set_index('time', inplace=True)

# # Resample the DataFrame to represent data sampled every 0.04 seconds
# df_resampled = df_eda.resample('40L').mean()

# # Reset the index to have 'time' as a regular column again
# df_resampled.reset_index(inplace=True)

# # Print the head of the resampled DataFrame
# print(df_resampled.head())


# df_merged = pd.merge(df_valence, df_eda, on="time", how="inner")

# # Read the CSV file into a DataFrame
# df_biosignals = pd.read_csv(csv_path_biosignals, delimiter=';')
# df_valence = pd.read_csv(csv_path_valence, delimiter=';')
# df_arousal = pd.read_csv(csv_path_arousal, delimiter=';')