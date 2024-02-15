import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import os

# Step 1: Read the CSV file
project_dir = os.path.dirname(os.path.abspath(__file__))
#file = "randomized_merged_dataset.csv"
file = "merged_dataset.csv"
csv_path = os.path.join(project_dir, '../', "dataset_csv_eda_arousal", file)
data = pd.read_csv(csv_path, delimiter=';')

# Step 2: Preprocess the data (if needed)

# Step 3: Split the data into features (X) and target variable (y)
X = data[['EDA', 'is_peak']]  # Features
y = data['classes']  # Target variable

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import classification_report
# from sklearn.model_selection import ShuffleSplit, cross_val_score
# import os

# # Step 1: Read the CSV file
# project_dir = os.path.dirname(os.path.abspath(__file__))
# file = "merged_dataset.csv"
# csv_path = os.path.join(project_dir, '../', "dataset_csv_eda_arousal", file)
# data = pd.read_csv(csv_path, delimiter=';')

# # Step 2: Preprocess the data (if needed)

# # Step 3: Split the data into features (X) and target variable (y)
# X = data[['EDA', 'is_peak']]  # Features
# y = data['classes']  # Target variable

# # Step 4: Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 5: Train the model
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)

# # Step 6: Evaluate the model
# y_pred = model.predict(X_test)
# print("Classification Report without Cross-Validation:")
# print(classification_report(y_test, y_pred))

# # Step 7: Cross-validation
# clf = DecisionTreeClassifier(random_state=42)
# ss = ShuffleSplit(train_size=0.7, test_size=0.2, n_splits=5)
# scores = cross_val_score(clf, X, y, cv=ss)

# print("\nCross Validation Scores: ", scores)
# print("Average CV Score: ", scores.mean())
# print("Number of CV Scores used in Average: ", len(scores))

##########################
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import classification_report
# from sklearn.model_selection import LeavePOut, cross_val_score
# import os

# # Step 1: Read the CSV file
# project_dir = os.path.dirname(os.path.abspath(__file__))
# file = "merged_dataset.csv"
# csv_path = os.path.join(project_dir, '../', "dataset_csv_eda_arousal", file)
# data = pd.read_csv(csv_path, delimiter=';')

# # Step 2: Preprocess the data (if needed)

# # Step 3: Split the data into features (X) and target variable (y)
# X = data[['EDA', 'is_peak']]  # Features
# y = data['classes']  # Target variable

# # Step 4: Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 5: Train the model
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)

# # Step 6: Evaluate the model
# y_pred = model.predict(X_test)
# print("Classification Report without Cross-Validation:")
# print(classification_report(y_test, y_pred))

# # Step 7: Cross-validation
# clf = DecisionTreeClassifier(random_state=42)
# lpo = LeavePOut(p=2)
# scores = cross_val_score(clf, X, y, cv=lpo)

# print("\nCross Validation Scores: ", scores)
# print("Average CV Score: ", scores.mean())
# print("Number of CV Scores used in Average: ", len(scores))


#

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import classification_report
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# import os

# # Step 1: Read the CSV file
# project_dir = os.path.dirname(os.path.abspath(__file__))
# file = "merged_dataset.csv"
# csv_path = os.path.join(project_dir, '../', "dataset_csv_eda_arousal", file)
# data = pd.read_csv(csv_path, delimiter=';')

# # Step 2: Preprocess the data (if needed)

# # Step 3: Split the data into features (X) and target variable (y)
# X = data[['EDA', 'is_peak']]  # Features
# y = data['classes']  # Target variable

# # Step 4: Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 5: Train the model
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)

# # Step 6: Evaluate the model
# y_pred = model.predict(X_test)
# print("Classification Report without Cross-Validation:")
# print(classification_report(y_test, y_pred))

# # Step 7: Cross-validation
# clf = DecisionTreeClassifier(random_state=42)
# sk_folds = StratifiedKFold(n_splits=3)
# scores = cross_val_score(clf, X, y, cv=sk_folds)

# print("\nCross Validation Scores: ", scores)
# print("Average CV Score: ", scores.mean())
# print("Number of CV Scores used in Average: ", len(scores))
