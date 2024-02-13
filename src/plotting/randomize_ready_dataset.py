#randomize
# import pandas as pd
# import os

# project_dir = os.path.dirname(os.path.abspath(__file__))
# print(project_dir)

# file = "merged_dataset.csv"
# # Load the CSV file using the project directory
# csv_path_valence = os.path.join(project_dir, "dataset_csv_eda_valence", file)


# # Load the CSV file into a DataFrame
# df = pd.read_csv(csv_path_valence, delimiter=';')

# # Shuffle the rows randomly
# df = df.sample(frac=1).reset_index(drop=True)

# # Save the randomized DataFrame back to a CSV file
# output_csv_path = os.path.join(project_dir, "dataset_csv_eda_valence", "randomized_merged_dataset.csv") #dataset_csv_eda_valence dataset_csv_eda_arousal

# df.to_csv(output_csv_path, index=False, sep=';')

# print("Rows randomized successfully.")

#multipliy by 3

import pandas as pd
import os

project_dir = os.path.dirname(os.path.abspath(__file__))
print(project_dir)

file = "merged_dataset.csv"
# Load the CSV file using the project directory
csv_path_valence = os.path.join(project_dir, "dataset_csv_eda_valence", file)

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_path_valence, delimiter=';')

# Create duplicates of each row
df_duplicates = pd.concat([df] * 3, ignore_index=True)
df_duplicates = df_duplicates.sample(frac=1).reset_index(drop=True)

# Save the DataFrame with duplicates back to a CSV file
output_csv_path = os.path.join(project_dir, "dataset_csv_eda_valence", "triplicated_merged_dataset.csv")
df_duplicates.to_csv(output_csv_path, index=False, sep=';')

print("Rows triplicated successfully.")
