import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import os
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, LeaveOneOut


# Read the first CSV file for EDA features
project_dir_eda = os.path.dirname(os.path.abspath(__file__))
file_eda = "merged_dataset.csv"
csv_path_eda = os.path.join(project_dir_eda, '../', "dataset_csv_eda_arousal", file_eda)
data_eda = pd.read_csv(csv_path_eda, delimiter=';')

# Read the second CSV file for ECG features
project_dir_ecg = os.path.dirname(os.path.abspath(__file__))
file_ecg = "merged_dataset.csv"
csv_path_ecg = os.path.join(project_dir_ecg, '../', "dataset_csv_ecg_arousal", file_ecg)
data_ecg = pd.read_csv(csv_path_ecg, delimiter=';')

# Combine features from both datasets
X_eda = data_eda[['EDA', 'is_peak']]  # EDA features
X_ecg = data_ecg[['ECG ', 'is_high_frequency']]  # ECG features
X_combined = pd.concat([X_eda, X_ecg], axis=1)  # Combined features
y = data_eda['classes']  # Target variable (assuming both datasets have the same target variable)

# Split the combined data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=71)

# Train the KNN classifier model
model = KNeighborsClassifier(algorithm='auto')
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))



# k_folds = KFold(n_splits = 100)
# sk_folds = StratifiedKFold(n_splits=100)

# scores = cross_val_score(model, X_combined, y, cv = k_folds)
# print("k_folds")
# print("Cross Validation Scores: ", scores)
# print("Average CV Score: ", scores.mean())
# print("Number of CV Scores used in Average: ", len(scores))


# scores = cross_val_score(model, X_combined, y, cv = sk_folds)
# print("sk_folds")
# print("Cross Validation Scores: ", scores)
# print("Average CV Score: ", scores.mean())
# print("Number of CV Scores used in Average: ", len(scores))


# loo = LeaveOneOut()

# scores = cross_val_score(model, X_combined, y, cv = loo)
# print("loo")
# print("Cross Validation Scores: ", scores)
# print("Average CV Score: ", scores.mean())
# print("Number of CV Scores used in Average: ", len(scores))