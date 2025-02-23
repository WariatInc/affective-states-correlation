import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import neurokit2 as nk
import numpy as np
import argparse
# Determine project directory
project_dir = os.path.dirname(os.path.abspath(__file__))
print(project_dir)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('file', metavar='F', type=str, help='the CSV file to process')
args = parser.parse_args()

# Determine project directory
project_dir = os.path.dirname(os.path.abspath(__file__))
print(project_dir)
print(args.file)
file = args.file
#file = "P16.csv"
# Load the CSV file using the project directory
csv_path_biosignals = os.path.join(project_dir, "../datasets/RECOLA-Biosignals-recordings", file)
csv_path_valence = os.path.join(project_dir, "../datasets/RECOLA-Annotation", "emotional_behaviour", "valence", file)
csv_path_arousal = os.path.join(project_dir, "../datasets/RECOLA-Annotation", "emotional_behaviour", "arousal", file)

# Read the CSV file into a DataFrame
df_biosignals = pd.read_csv(csv_path_biosignals, delimiter=';')
df_valence = pd.read_csv(csv_path_valence, delimiter=';')
df_arousal = pd.read_csv(csv_path_arousal, delimiter=';')

df_biosignals.columns = df_biosignals.columns.str.strip()
df_valence.columns = df_valence.columns.str.strip()
df_arousal.columns = df_arousal.columns.str.strip()

df_biosignals = df_biosignals[df_biosignals['time'] <= 300]
df_valence = df_valence[df_valence['time'] <= 300]
df_arousal = df_arousal[df_arousal['time'] <= 300]


# Changing the frequency of sampling - 1 is standard
df_biosignals = df_biosignals.set_index('time').rolling(window=1).mean().dropna()[
                ::1]  # normally biosignals sample rate is every 0.001s
df_valence = df_valence.set_index('time').rolling(window=1).mean().dropna()[
             ::1]  # normally behaviour sample rate is every 0.04s
df_arousal = df_arousal.set_index('time').rolling(window=1).mean().dropna()[
             ::1]  # normally behaviour sample rate is every 0.04s


def max_abs_value(row):
    return max(row.min(), row.max(), key=abs)


# Calculate the values from 'FM1', 'FM2', 'FM3', 'FF1', 'FF2', and 'FF3' columns
df_valence['mean_behaviour_rate'] = df_valence[['FM1', 'FM2', 'FM3', 'FF1', 'FF2', 'FF3']].mean(axis=1)
df_valence['max_abs_behaviour_rate'] = df_valence[['FM1', 'FM2', 'FM3', 'FF1', 'FF2', 'FF3']].abs().max(axis=1)
df_valence['max_from_0_behaviour_rate'] = df_valence[['FM1', 'FM2', 'FM3', 'FF1', 'FF2', 'FF3']].apply(max_abs_value,
                                                                                                       axis=1)

df_arousal['mean_behaviour_rate'] = df_arousal[['FM1', 'FM2', 'FM3', 'FF1', 'FF2', 'FF3']].mean(axis=1)
df_arousal['max_abs_behaviour_rate'] = df_arousal[['FM1', 'FM2', 'FM3', 'FF1', 'FF2', 'FF3']].abs().max(axis=1)
df_arousal['max_from_0_behaviour_rate'] = df_arousal[['FM1', 'FM2', 'FM3', 'FF1', 'FF2', 'FF3']].apply(max_abs_value,
                                                                                                       axis=1)

# mean_eda_value = df_biosignals['EDA'].mean()

# print("Mean EDA value:", mean_eda_value)
# df_biosignals['EDA'] -= mean_eda_value
df_biosignals['ECG_mirror'] = df_biosignals['ECG'] * -1 * -1

df_biosignals['ECG_smooth'] = df_biosignals['ECG_mirror'].rolling(window=1000, min_periods=1).mean()
df_biosignals['EDA_smooth'] = df_biosignals['EDA'].rolling(window=10, min_periods=1).mean()

# ====================== lukas code ======================

# Normalization of EDA
original_series = df_biosignals['EDA_smooth']
normalized_series = 0.5*(original_series - original_series.min()) / (original_series.max() - original_series.min())
df_biosignals['EDA_smooth'] = normalized_series

# Normalization of ECG
original_series = df_biosignals['ECG_smooth']
normalized_series = (original_series - original_series.min()) / (original_series.max() - original_series.min()) - 0.5
df_biosignals['ECG_smooth'] = normalized_series

# # Normalization of AROUSAL
# original_series = df_arousal['mean_behaviour_rate']
# normalized_series = (original_series - original_series.min()) / (original_series.max() - original_series.min()) - 0.5
# df_arousal['mean_behaviour_rate'] = normalized_series
#
# # Normalization of VALANCE
# original_series = df_valence['mean_behaviour_rate']
# normalized_series = (original_series - original_series.min()) / (original_series.max() - original_series.min()) - 0.5
# df_valence['n_mean_behaviour_rate'] = normalized_series

# Find peaks
eda_smooth_series = df_biosignals['ECG_smooth']

#Find peaks using scipy's find_peaks function
peaks, _ = find_peaks(x=eda_smooth_series, distance=500, width=20)
min_peaks, _ = find_peaks(x=-eda_smooth_series, distance=500, width=20)
#print(peaks)
# # Find peaks using neurokit2 find_peaks function
# _, nk_data = nk.eda_peaks(eda_smooth_series)
# peaks = nk_data['SCR_Peaks']

# # Define the number of neighboring points to mark around each peak
# neighborhood = 1000

# # Create an empty list to store the extended peaks
# extended_peaks = []

# # Iterate over each detected peak
# for peak_index in peaks:
#     # Extend the peak by including neighboring points
#     extended_peak_indices = np.arange(max(peak_index - neighborhood, 0), min(peak_index + neighborhood + 1, len(eda_smooth_series)))
#     extended_peaks.extend(extended_peak_indices)

# # Remove duplicate indices
# extended_peaks = list(set(extended_peaks))

# Plot peaks scatter points and lines
# plt.scatter(df_biosignals['EDA_smooth'].index[extended_peaks], eda_smooth_series.iloc[peaks], color='red', label='Peaks')
# # plt.scatter(df_biosignals['EDA_smooth'].index[min_peaks], eda_smooth_series.iloc[min_peaks], color='green', label='Min Peaks')
# plt.vlines(df_biosignals['EDA_smooth'].index[peaks], ymin=-0.2, ymax=0.5, colors='purple', linestyles='dashed', label='Vertical Lines at -0.5 and 0.5')

#plt.scatter(df_biosignals['ECG_smooth'].index[extended_peaks], eda_smooth_series.iloc[extended_peaks], color='blue', label='Extended Peaks')
# time_intervals = np.diff(df_biosignals['ECG_smooth'].index[peaks])

# # Convert time intervals to RR intervals (in seconds)
# rr_intervals = time_intervals / df_biosignals.index.freq.seconds

# # Calculate time intervals between successive peaks
time_intervals = np.diff(df_biosignals.index[peaks])

# Convert time intervals to RR intervals (in seconds)
rr_intervals = time_intervals
rr_intervals = np.append(rr_intervals, rr_intervals[-1])

frequency = (1/rr_intervals * 60 )
#print(frequency)
# Now rr_intervals contains the RR intervals
print("RR Intervals (in seconds):", frequency)

plt.scatter(df_biosignals['ECG_smooth'].index[peaks], eda_smooth_series.iloc[peaks], color='red', label='ECG Spikes')
# Plot the associated values
def fit_polynomial(x_values, y_values):
    trend_lines = []
    for i in range(len(x_values)):
        # Take the nearest 3 samples
        start_idx = max(0, i -1)
        end_idx = min(len(x_values), i +2)
        x_window = x_values[start_idx:end_idx]
        y_window = y_values[start_idx:end_idx]

        # Fit a polynomial curve to the window
        trend_line = np.poly1d(np.polyfit(x_window, y_window, deg=2))
        trend_lines.append(trend_line(x_values[i]))
       # df_biosignals['trend_lines'] = trend_lines
    return trend_lines

# Fit the polynomial curve using the nearest 3 samples
frequency_line = fit_polynomial(df_biosignals['ECG_smooth'].index[peaks], frequency)
resampled_frequency_line = np.interp(df_biosignals.index, df_biosignals['ECG_smooth'].index[peaks], frequency_line)

# Add the resampled smoothed values as a new column to the existing DataFrame
df_biosignals['frequency'] = resampled_frequency_line
plt.plot(df_biosignals.index, df_biosignals['frequency'], label='frequency heartbeat')  #PLOT FREQUENCY
# Plot the scatter plot of the data points 

plt.scatter(df_biosignals['ECG_smooth'].index[peaks], frequency, color='pink', label='FREQUENCY POINTS') #PLOT FREQUENCY POINTS


frequency_column = df_biosignals['frequency']
mean_of_column_frequency = df_biosignals['frequency'].mean()

high_frequency_points = df_biosignals['frequency'][df_biosignals['frequency'] > mean_of_column_frequency*1]

high_frequency_indices = np.array(high_frequency_points.index)
#Plot the high-frequency points
plt.scatter(high_frequency_points.index, high_frequency_points, color='magenta', label='High Frequency Points')

high_frequency_points = list(set(high_frequency_points))
print(high_frequency_indices)
# Find peaks using neurokit2 find_peaks function#####################################################################
# _, nk_data = nk.eda_peaks(frequency_column)
# peaks = nk_data['SCR_Peaks']

# # Define the number of neighboring points to mark around each peak
# neighborhood = 2000

# # Create an empty list to store the extended peaks
# extended_peaks = []

# # Iterate over each detected peak
# for peak_index in peaks:
#     # Extend the peak by including neighboring points
#     extended_peak_indices = np.arange(max(peak_index - neighborhood, 0), min(peak_index + neighborhood + 1, len(frequency_column)))
#     extended_peaks.extend(extended_peak_indices)

# # Remove duplicate indices
# extended_peaks = list(set(extended_peaks))
# plt.scatter(df_biosignals['frequency'].index[extended_peaks], frequency_column.iloc[extended_peaks], color='navy', label='Extended Peaks')
# plt.scatter(df_biosignals['frequency'].index[peaks], frequency_column.iloc[peaks], color='brown', label='Original Peaks')
################################################################################################
# Plot the smoothed trend line
#plt.plot(df_biosignals['ECG_smooth'].index[peaks], smooth_line_y, color='blue', label='Smoothed Trend Line')


# ========================================================

def mark_peaks_and_save_to_csv_ECG(project_dir, file, high_frequency_points):
    # df_all_indices = pd.DataFrame({'time': range(len(df_biosignals))})

    # # Mark whether each index corresponds to a HIGH FREQUENCY OR NOT
    # df_all_indices['is_high_frequency'] = df_all_indices['time'].isin(high_frequency_points)

    # # Load the CSV file using the project directory
    # csv_path_biosignals = os.path.join(project_dir, "RECOLA-Biosignals-recordings", file)

    # Read the CSV file into a DataFrame
    df_biosignals2 = pd.read_csv(csv_path_biosignals, delimiter=';')

    # Define the output CSV path
    output_csv_path = os.path.join(project_dir, "../datasets/ECG_peaks_determined_csv", f"{file}_peaks.csv")

    # # Concatenate the 'is_peak' column with df_biosignals2
    # df_biosignals2['is_high_frequency'] = df_all_indices['is_high_frequency']

    # Drop the 'ECG' column
    df_biosignals2.drop(columns=['EDA'], inplace=True)

    df_biosignals2['is_high_frequency'] = False

    # Mark rows as True where 'time' matches high frequency indices
    df_biosignals2.loc[df_biosignals2['time'].isin(high_frequency_indices), 'is_high_frequency'] = True


    # Save the concatenated DataFrame to a CSV file with delimiter ';'
    df_biosignals2.to_csv(output_csv_path, index=False, sep=';')

#mark_peaks_and_save_to_csv_ECG(project_dir, file, high_frequency_points)



plt.plot(df_biosignals.index, df_biosignals['ECG_smooth'], label='ECG_smooth')
#plt.plot(df_biosignals.index, df_biosignals['EDA_smooth'], label='EDA_smooth')


#plt.plot(df_biosignals.index, df_biosignals['EDA'], label='EDA')
#plt.plot(df_biosignals.index, df_biosignals['ECG'], label='ECG')

# Plot the data from the second CSV file - valence
# plt.plot(df_valence.index, df_valence['FM1'], label='valence_FM1')
# plt.plot(df_valence.index, df_valence['FM2'], label='valence_FM2')
# plt.plot(df_valence.index, df_valence['FM3'], label='valence_FM3')
# plt.plot(df_valence.index, df_valence['FF1'], label='valence_FF1')
# plt.plot(df_valence.index, df_valence['FF2'], label='valence_FF2')
# plt.plot(df_valence.index, df_valence['FF3'], label='valence_FF3')

# Plot the data from the second CSV file - arousal
# plt.plot(df_arousal.index, df_arousal['FM1'], label='arousal_FM1')
# plt.plot(df_arousal.index, df_arousal['FM2'], label='arousal_FM2')
# plt.plot(df_arousal.index, df_arousal['FM3'], label='arousal_FM3')
# plt.plot(df_arousal.index, df_arousal['FF1'], label='arousal_FF1')
# plt.plot(df_arousal.index, df_arousal['FF2'], label='arousal_FF2')
# plt.plot(df_arousal.index, df_arousal['FF3'], label='arousal_FF3')

#
plt.plot(df_valence.index, df_valence['mean_behaviour_rate'], label='valence_mean_behaviour_rate')
# plt.plot(df_valence.index, df_valence['max_abs_behaviour_rate'], label='valence_max_abs_behaviour_rate')
# plt.plot(df_valence.index, df_valence['max_from_0_behaviour_rate'], label='valence_max_from_0_behaviour_rate')

#plt.plot(df_arousal.index, df_arousal['mean_behaviour_rate'], label='arousal_mean_behaviour_rate')
# plt.plot(df_arousal.index, df_arousal['max_abs_behaviour_rate'], label='arousal_max_abs_behaviour_rate')
# plt.plot(df_arousal.index, df_arousal['max_from_0_behaviour_rate'], label='arousal_max_from_0_behaviour_rate')



plt.xlabel('Time')
plt.ylabel('Values')
plt.title('EDA and ECG Plot')
plt.legend()
plt.grid(True)



## ============================== DRAFT ==============================
#Derivative
# eda_smooth_series = df_biosignals['EDA_smooth']
# ecg_smooth_series = df_biosignals['ECG_smooth']
# ecg_derivative = np.gradient(ecg_smooth_series, df_biosignals.index)

# rolling_mean = pd.Series(ecg_derivative).rolling(window=1000, min_periods=5).mean()
# normalized_derivative = (rolling_mean - rolling_mean.min()) / (rolling_mean.max() - rolling_mean.min()) - 0.5
# plt.plot(df_biosignals.index, normalized_derivative, label='Derivative', linestyle='--')

#plt.show()
#python src/plotting_scripts/modify_csv_ecg.py P16.csv && python src/plotting_scripts/modify_csv_ecg.py P19.csv && python src/plotting_scripts/modify_csv_ecg.py P21.csv && python src/plotting_scripts/modify_csv_ecg.py P23.csv && python src/plotting_scripts/modify_csv_ecg.py P25.csv && python src/plotting_scripts/modify_csv_ecg.py P26.csv && python src/plotting_scripts/modify_csv_ecg.py P28.csv && python src/plotting_scripts/modify_csv_ecg.py P30.csv && python src/plotting_scripts/modify_csv_ecg.py P34.csv && python src/plotting_scripts/modify_csv_ecg.py P37.csv && python src/plotting_scripts/modify_csv_ecg.py P39.csv && python src/plotting_scripts/modify_csv_ecg.py P41.csv && python src/plotting_scripts/modify_csv_ecg.py P42.csv && python src/plotting_scripts/modify_csv_ecg.py P45.csv && python src/plotting_scripts/modify_csv_ecg.py P46.csv && python src/plotting_scripts/modify_csv_ecg.py P56.csv && python src/plotting_scripts/modify_csv_ecg.py P64.csv && python src/plotting_scripts/modify_csv_ecg.py P65.csv
