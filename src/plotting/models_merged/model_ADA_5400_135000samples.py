import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold

# Read the first CSV file for ECG features
project_dir_ecg = os.path.dirname(os.path.abspath(__file__))
file_ecg = "merged_dataset.csv"
csv_path_ecg = os.path.join(project_dir_ecg, '../', "dataset_csv_ecg_arousal", file_ecg)
data_ecg = pd.read_csv(csv_path_ecg, delimiter=';')

# Read the second CSV file for EDA features
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

# Train the AdaBoost model
adaboost = AdaBoostClassifier(n_estimators=500, algorithm='SAMME', random_state=42)
adaboost.fit(X_train, y_train)

# Evaluate the model
y_pred = adaboost.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
