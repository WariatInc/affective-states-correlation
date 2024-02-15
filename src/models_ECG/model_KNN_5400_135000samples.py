#
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import os

# Step 1: Read the CSV file
project_dir = os.path.dirname(os.path.abspath(__file__))
file = "merged_dataset.csv"
# Load the CSV file using the project directory
csv_path = os.path.join(project_dir, '../datasets/', "dataset_csv_ecg_valence", file)
data = pd.read_csv(csv_path, delimiter=';')


# Step 2: Preprocess the data (if needed)

# Step 3: Split the data into features (X) and target variable (y)
X = data[['ECG ', 'is_high_frequency']]  # Features#
y = data['classes']  # Target variable
# Step 2: Preprocess the data (if needed)

# Step 3: Split the data into features (X) and target variable (y)
X = data[['ECG ', 'is_high_frequency']]  # Features#
y = data['classes']  # Target variable
# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=71)

# Step 5: Train the model
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
