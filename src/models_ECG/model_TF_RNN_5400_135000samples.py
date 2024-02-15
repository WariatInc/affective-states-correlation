import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

# Step 1: Read the CSV file
project_dir = os.path.dirname(os.path.abspath(__file__))
print(project_dir)
 # Assuming you're running this script from the same directory where the CSV file is located
file = "merged_dataset.csv"
csv_path = os.path.join(project_dir, '../datasets/', "dataset_csv_eda_valence", file)
data = pd.read_csv(csv_path, delimiter=';')

# Step 2: Preprocess the data (if needed)

# Step 3: Split the data into features (X) and target variable (y)
X = data[['EDA', 'is_peak']].astype('float32').values  # Features
y = data['classes'].astype('int32').values  # Target variable

# Step 4: Handle NaN and infinity values
X[np.isnan(X)] = np.nanmean(X)  # Replace NaN with the mean
X[np.isinf(X)] = 0  # Replace infinity with 0

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Reshape the data for RNN input
# Reshape X to (samples, time_steps, features)
X_train = X_train.reshape(-1, 1, 2)
X_test = X_test.reshape(-1, 1, 2)

# Step 7: Define the RNN architecture
model = Sequential()
model.add(LSTM(64, input_shape=(1, 2)))  # LSTM layer with 64 units
model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification

# Step 8: Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 9: Train the model
model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_test, y_test))

# Step 10: Evaluate the model
predict_x = model.predict(X_test)
classes_x = np.where(predict_x > 0.5, 1, 0)  # Convert probabilities to class labels

print(classification_report(y_test, classes_x))
