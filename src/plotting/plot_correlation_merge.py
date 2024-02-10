import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Determine project directory
project_dir = os.path.dirname(os.path.abspath(__file__))
print(project_dir)

csv_path_valence = os.path.join(project_dir, "RECOLA-Annotation", "emotional_behaviour", "valence")
csv_path_list_valence = glob.glob(csv_path_valence + '/*')

csv_path_arousal = os.path.join(project_dir, "RECOLA-Annotation", "emotional_behaviour", "arousal")
csv_path_list_arousal = glob.glob(csv_path_arousal + '/*')

# Load the CSV file using the project directory


# Read the CSV file into a DataFrame
df_valence_mean_merge = pd.DataFrame()
df_arousal_mean_merge = pd.DataFrame()

for i, path in enumerate(csv_path_list_valence):
    df_valence = pd.read_csv(path, delimiter=';')
    df_valence.columns = df_valence.columns.str.strip()
    df_valence = df_valence[df_valence['time'] <= 300]
    df_valence = df_valence.set_index('time').rolling(window=1).mean().dropna()[
                 ::1]  # normally behaviour sample rate is every 0.04s
    new_series = pd.Series(df_valence[['FM1', 'FM2', 'FM3', 'FF1', 'FF2', 'FF3']].mean(axis=1), name=f'mean_{i}')
    df_valence_mean_merge = df_valence_mean_merge.append(new_series)

df_valence_mean_merge = df_valence_mean_merge.T
df_valence_mean_merge['mean_behaviour_rate'] = df_valence_mean_merge[df_valence_mean_merge.columns].mean(axis=1)
total_valence_mean = df_valence_mean_merge['mean_behaviour_rate'].mean()
min_valence_mean = df_valence_mean_merge['mean_behaviour_rate'].max()
max_valence_mean = df_valence_mean_merge['mean_behaviour_rate'].min()

for i, path in enumerate(csv_path_list_arousal):
    df_arousal = pd.read_csv(path, delimiter=';')
    df_arousal.columns = df_arousal.columns.str.strip()
    df_arousal = df_arousal[df_arousal['time'] <= 300]
    df_arousal = df_arousal.set_index('time').rolling(window=1).mean().dropna()[
                 ::1]  # normally behaviour sample rate is every 0.04s
    new_series = pd.Series(df_arousal[['FM1', 'FM2', 'FM3', 'FF1', 'FF2', 'FF3']].mean(axis=1), name=f'mean_{i}')
    df_arousal_mean_merge = df_arousal_mean_merge.append(new_series)

df_arousal_mean_merge = df_arousal_mean_merge.T
df_arousal_mean_merge['mean_behaviour_rate'] = df_arousal_mean_merge[df_arousal_mean_merge.columns].mean(axis=1)
total_arousal_mean = df_arousal_mean_merge['mean_behaviour_rate'].mean()
max_arousal_mean = df_arousal_mean_merge['mean_behaviour_rate'].max()
min_arousal_mean = df_arousal_mean_merge['mean_behaviour_rate'].min()

# print("Mean EDA value:", mean_eda_value)
# df_biosignals['EDA'] -= mean_eda_value

# # Valance
plt.figure(1)
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('The mean valance of all examinees in time')

plt.grid(True)
plt.plot(df_valence_mean_merge.index, df_valence_mean_merge['mean_behaviour_rate'], label='valence_mean_behaviour_rate',
         linewidth=4)
plt.axhline(y=total_valence_mean, color='red', linestyle='--', label='The total mean valance')
for i in range(10):
    plt.plot(df_valence_mean_merge.index, df_valence_mean_merge[f'mean_{i}'], linewidth=0.3)

plt.text(x=1, y=0.6, ha='left', va='bottom', s=f'TotalMeanValence = {total_valence_mean:.3f}', fontsize=12)
plt.text(x=1, y=0.55, ha='left', va='bottom', s=f'MaxMeanValence = {min_valence_mean:.3f}', fontsize=12)
plt.text(x=1, y=0.5, ha='left', va='bottom', s=f'MinMeanValence = {max_valence_mean:.3f}', fontsize=12)
plt.legend(loc='upper right')

# Arousal
plt.figure(2)
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('The mean arousal of all examinees in time')
plt.grid(True)
plt.plot(df_arousal_mean_merge.index, df_arousal_mean_merge['mean_behaviour_rate'], label='arousal_mean_behaviour_rate',
         linewidth=4)
plt.axhline(y=total_arousal_mean, color='red', linestyle='--', label='The total mean arousal')
for i in range(10):
    plt.plot(df_arousal_mean_merge.index, df_arousal_mean_merge[f'mean_{i}'], linewidth=0.3)

plt.text(x=1, y=0.4, ha='left', va='bottom', s=f'TotalMeanArousal = {total_arousal_mean:.3f}', fontsize=12)
plt.text(x=1, y=0.35, ha='left', va='bottom', s=f'MaxMeanArousal = {max_arousal_mean:.3f}', fontsize=12)
plt.text(x=1, y=0.3, ha='left', va='bottom', s=f'MinMeanArousal = {min_arousal_mean:.3f}', fontsize=12)
plt.legend(loc='upper right')

plt.show()
