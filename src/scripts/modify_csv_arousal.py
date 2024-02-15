import os
import pandas as pd
import matplotlib.pyplot as plt

#valence values should be renamed to arousal 

def add_classes_and_save_to_csv(project_dir, file):
    # Load the CSV file using the project directory
    csv_path_valence = os.path.join(project_dir, "../datasets/RECOLA-Annotation", "emotional_behaviour", "arousal", file)

    # Read the CSV file into a DataFrame
    df_valence = pd.read_csv(csv_path_valence, delimiter=';')

    # Clean column names
    df_valence.columns = df_valence.columns.str.strip()

    # Filter data
    df_valence = df_valence[df_valence['time'] <= 300]

    # Calculate mean behaviour rate
    df_valence['mean_behaviour_rate'] = df_valence[['FM1', 'FM2', 'FM3', 'FF1', 'FF2', 'FF3']].mean(axis=1)

    # Define the bins and labels for the class
    top_neutral_border = 0.00000001
    bottom_neutral_border = 0.00000000999
    bins = [-float('inf'), bottom_neutral_border, top_neutral_border, float('inf')]
    labels = [-1, 0, 1]
    df_valence['classes'] = pd.cut(df_valence['mean_behaviour_rate'], bins=bins, labels=labels, include_lowest=True)

    # Define the output CSV path
    output_csv_path = os.path.join(project_dir, "../datasets/arousal_classes_csv", file)

    # Save the DataFrame to a CSV file
    df_valence.to_csv(output_csv_path, index=False, sep=';')

# Determine project directory
project_dir = os.path.dirname(os.path.abspath(__file__))

# List of files
files = ["P16.csv", "P19.csv", "P21.csv", "P23.csv", "P25.csv", "P26.csv", "P28.csv", "P30.csv", "P34.csv", "P37.csv", "P39.csv", "P41.csv", "P42.csv", "P45.csv", "P46.csv", "P56.csv", "P64.csv", "P65.csv"]

# Loop through each file
for file in files:
    add_classes_and_save_to_csv(project_dir, file)
