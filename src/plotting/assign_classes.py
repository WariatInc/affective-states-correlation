import os
import pandas as pd
import matplotlib.pyplot as plt

# Determine project directory
project_dir = os.path.dirname(os.path.abspath(__file__))
print(project_dir)

file = "P19.csv"

# Load the CSV file using the project directory
csv_path_valence = os.path.join(project_dir, "RECOLA-Annotation", "emotional_behaviour", "valence", file)
csv_path_arousal = os.path.join(project_dir, "RECOLA-Annotation", "emotional_behaviour", "arousal", file)

# Read the CSV file into a DataFrame
df_valence = pd.read_csv(csv_path_valence, delimiter=';')
df_arousal = pd.read_csv(csv_path_arousal, delimiter=';')

df_valence.columns = df_valence.columns.str.strip()
df_arousal.columns = df_arousal.columns.str.strip()

df_valence = df_valence[df_valence['time'] <= 300]
df_arousal = df_arousal[df_arousal['time'] <= 300]

df_valence = df_valence.set_index('time').rolling(window=1).mean().dropna()[
             ::1]  # normally behaviour sample rate is every 0.04s
df_arousal = df_arousal.set_index('time').rolling(window=1).mean().dropna()[
             ::1]  # normally behaviour sample rate is every 0.04s

df_valence['mean_behaviour_rate'] = df_valence[['FM1', 'FM2', 'FM3', 'FF1', 'FF2', 'FF3']].mean(axis=1)
df_arousal['mean_behaviour_rate'] = df_arousal[['FM1', 'FM2', 'FM3', 'FF1', 'FF2', 'FF3']].mean(axis=1)

# Define the bins and labels for the class
top_neutral_border = 0.15
bottom_neutral_border = 0.05
bins = [-float('inf'), bottom_neutral_border, top_neutral_border, float('inf')]
labels = [-1, 0, 1]
df_valence['classes'] = pd.cut(df_valence['mean_behaviour_rate'], bins=bins, labels=labels, include_lowest=True)

# # Valance
plt.figure(1)
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('The mean valance of all examinees in time')

plt.grid(True)
# plt.vlines(x=df_valence.index, ymin=0, ymax=df_valence['classes'], linewidth=1, color='blue', alpha=0.3, label='valence_mean_behaviour_rate')
plt.vlines(x=df_valence[df_valence['classes'] == 1].index, ymin=top_neutral_border, ymax=df_valence[df_valence['classes'] == 1]['mean_behaviour_rate'],
           linewidth=1, color='pink', alpha=0.5, label='Class 1 - high valence')

plt.vlines(x=df_valence[df_valence['classes'] == -1].index, ymax=bottom_neutral_border, ymin=df_valence[df_valence['classes'] == -1]['mean_behaviour_rate'],
           linewidth=1, color='grey', alpha=0.5, label='Class -1 - low valence')

plt.vlines(x=df_valence[df_valence['classes'] == 0].index, ymin=bottom_neutral_border, ymax=df_valence[df_valence['classes'] == 0]['mean_behaviour_rate'],
           linewidth=1, color='blue', alpha=0.5, label='Class 0 - neutral valence')

# plt.hlines(y=0, xmin=df_valence.index.min(), xmax=df_valence.index.max(), linestyle='-',linewidth=2, color='blue', label='Horizontal Line when class is 0')

plt.plot(df_valence.index, df_valence['mean_behaviour_rate'], label='valence_mean_behaviour_rate', color='red',
         linewidth=1)
# plt.hlines(x=df_valence.index,y=df_valence['classes'], linewidth=3, color='red',
#            label='valence_mean_behaviour_rate')

plt.legend(loc='upper right')
plt.show()
