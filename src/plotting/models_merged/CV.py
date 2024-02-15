import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
import os
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
import numpy as np
# Read the first CSV file
project_dir_ecg = os.path.dirname(os.path.abspath(__file__))
file_ecg = "merged_dataset.csv"
csv_path_ecg = os.path.join(project_dir_ecg, '../', "dataset_csv_ecg_arousal", file_ecg)
data_ecg = pd.read_csv(csv_path_ecg, delimiter=';')

# Read the second CSV file
project_dir_eda = os.path.dirname(os.path.abspath(__file__))
file_eda = "merged_dataset.csv"
csv_path_eda = os.path.join(project_dir_eda, '../', "dataset_csv_eda_arousal", file_eda)
data_eda = pd.read_csv(csv_path_eda, delimiter=';')

# Preprocess the data (if needed)

# Combine features from both datasets
X_ecg = data_ecg[['ECG ', 'is_high_frequency']]  # ECG features
X_eda = data_eda[['EDA', 'is_peak']]  # EDA features
X_combined = pd.concat([X_ecg, X_eda], axis=1)  # Combined features
y = data_ecg['classes']  # Target variable (assuming both datasets have the same target variable)

# Split the combined data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)


# Assuming you have your data X and y
# X is your feature matrix
# y is your target variable

# Different max_depth values to test
max_depth_values = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

# Results dictionary to store average accuracies for each max_depth
results = {}

# Loop over max_depth values and perform cross-validation
for max_depth in max_depth_values:
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    cv_scores = cross_val_score(model, X_combined, y, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
    avg_accuracy = np.mean(cv_scores)
    results[max_depth] = avg_accuracy
    print(f"Max Depth: {max_depth}, Average Accuracy: {avg_accuracy}")

# Find the max_depth with the highest average accuracy
best_max_depth = max(results, key=results.get)
print(f"Best Max Depth: {best_max_depth}, Highest Average Accuracy: {results[best_max_depth]}")