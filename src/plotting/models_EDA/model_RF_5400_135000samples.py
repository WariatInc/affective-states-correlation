import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os


# Step 1: Read the CSV file

project_dir = os.path.dirname(os.path.abspath(__file__))
#project_dir = os.path.abspath(__file__)
print(project_dir)

file = "merged_dataset.csv"

# Load the CSV file using the project directory
csv_path = os.path.join(project_dir, '../', "dataset_csv_ecg_valence", file)
data = pd.read_csv(csv_path, delimiter=';')


# Step 2: Preprocess the data (if needed)

# Step 3: Split the data into features (X) and target variable (y)
X = data[['EDA', 'is_peak']]  # Features#
y = data['classes']  # Target variable

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=11)

# Step 5: Train the model
#model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model = RandomForestClassifier()

model.fit(X_train, y_train)
#z
# Step 6: Evaluate the model#
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


