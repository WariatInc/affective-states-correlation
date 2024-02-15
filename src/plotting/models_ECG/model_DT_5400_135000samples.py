import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import os

# Step 1: Read the CSV file
project_dir = os.path.dirname(os.path.abspath(__file__))
file = "merged_dataset.csv"
# Load the CSV file using the project directory
csv_path = os.path.join(project_dir, '../', "dataset_csv_ecg_valence", file)
data = pd.read_csv(csv_path, delimiter=';')


# Step 2: Preprocess the data (if needed)

# Step 3: Split the data into features (X) and target variable (y)
X = data[['ECG ', 'is_high_frequency']]  # Features#
y = data['classes']  # Target variable

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


######################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import ShuffleSplit, cross_val_score
import os

# Step 1: Read the CSV file
project_dir = os.path.dirname(os.path.abspath(__file__))
file = "merged_dataset.csv"
# Load the CSV file using the project directory
csv_path = os.path.join(project_dir, '../', "dataset_csv_ecg_valence", file)
data = pd.read_csv(csv_path, delimiter=';')


# Step 2: Preprocess the data (if needed)

# Step 3: Split the data into features (X) and target variable (y)
X = data[['ECG ', 'is_high_frequency']]  # Features
y = data['classes']  # Target variable

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report without Cross-Validation:")
print(classification_report(y_test, y_pred))

# Step 7: Cross-validation
clf = DecisionTreeClassifier(random_state=42)
ss = ShuffleSplit(train_size=0.6, test_size=0.3, n_splits=5)
scores = cross_val_score(clf, X, y, cv=ss)

print("\nCross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))








############



# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import classification_report
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# import os

# # Read the first CSV file
# project_dir_ecg = os.path.dirname(os.path.abspath(__file__))
# file_ecg = "merged_dataset.csv"
# csv_path_ecg = os.path.join(project_dir_ecg, '../', "dataset_csv_ecg_valence", file_ecg)
# data_ecg = pd.read_csv(csv_path_ecg, delimiter=';')

# # Read the second CSV file
# project_dir_eda = os.path.dirname(os.path.abspath(__file__))
# file_eda = "merged_dataset.csv"
# csv_path_eda = os.path.join(project_dir_eda, '../', "dataset_csv_eda_arousal", file_eda)
# data_eda = pd.read_csv(csv_path_eda, delimiter=';')

# # Preprocess the data (if needed)

# # Combine features from both datasets
# X_ecg = data_ecg[['ECG ', 'is_high_frequency']]  # ECG features
# X_eda = data_eda[['EDA', 'is_peak']]  # EDA features
# X_combined = pd.concat([X_ecg, X_eda], axis=1)  # Combined features
# y = data_ecg['classes']  # Target variable (assuming both datasets have the same target variable)

# # Split the combined data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# # Train the model
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)

# # Evaluate the model
# y_pred = model.predict(X_test)
# print("Classification Report:")
# print(classification_report(y_test, y_pred))
