import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report
import os

from sklearn.metrics import classification_report

# Step 1: Read the CSV file
project_dir = os.path.dirname(os.path.abspath(__file__))
file = "merged_dataset.csv"
csv_path = os.path.join(project_dir, '../datasets/', "dataset_csv_eda_valence", file)
data = pd.read_csv(csv_path, delimiter=';')

# Step 2: Preprocess the data (if needed)
# Split the data into features (X) and target variable (y)
X = data[['EDA', 'is_peak']]  # Features
y = data['classes']  # Target variable

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Build the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Step 5: Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Step 7: Evaluate the model
y_pred = model.predict_classes(X_test)
print(classification_report(y_test, y_pred))
