

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold


# Read the first CSV file for EDA features
project_dir_eda = os.path.dirname(os.path.abspath(__file__))
file_eda = "merged_dataset.csv"
csv_path_eda = os.path.join(project_dir_eda, '../datasets/', "dataset_csv_eda_valence", file_eda)
data_eda = pd.read_csv(csv_path_eda, delimiter=';')

# Read the second CSV file for ECG features
project_dir_ecg = os.path.dirname(os.path.abspath(__file__))
file_ecg = "merged_dataset.csv"
csv_path_ecg = os.path.join(project_dir_ecg, '../datasets/', "dataset_csv_ecg_valence", file_ecg)
data_ecg = pd.read_csv(csv_path_ecg, delimiter=';')

# Combine features from both datasets
X_eda = data_eda[['EDA', 'is_peak']]  # EDA features
X_ecg = data_ecg[['ECG ', 'is_high_frequency']]  # ECG features
X_combined = pd.concat([X_eda, X_ecg], axis=1)  # Combined features
y = data_eda['classes']  # Target variable (assuming both datasets have the same target variable)

# Split the combined data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Train the Gradient Boosting Classifier model
model = RandomForestClassifier(n_estimators=150, max_depth=21, random_state=27, verbose=1)
model.fit(X_train, y_train)







# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# k_folds = KFold(n_splits = 5)

# scores = cross_val_score(model, X_combined, y, cv = k_folds)

# print("Cross Validation Scores: ", scores)
# print("Average CV Score: ", scores.mean())
# print("Number of CV Scores used in Average: ", len(scores))
