import pandas as pd
import numpy as np
import os
import glob
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

project_dir = os.path.dirname(os.path.abspath(__file__))
print(project_dir)

csv_nn = os.path.join(project_dir,'../', "nn_dataset")
csv_nn_list = glob.glob(csv_nn + '/*')
NUM_CLASSES = 2

# Create a Normalization layer
def get_basic_model():
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(64, activation='relu', input_shape=(41,)),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Dense(32, activation='relu'),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    # ])

    # model = Sequential([
    #     Dense(64, activation='relu', input_shape=(41,)),
    #     Dense(64, activation='relu'),
    #     Dense(NUM_CLASSES, activation='sigmoid')
    # ])

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(41,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),  # Adjust the dropout rate as needed
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


# =================================NEW CODE ====================================
combined_df = pd.DataFrame(columns=['classes', 'is_peak', 'eda'])
for path in csv_nn_list:
    df = pd.read_csv(path, usecols=["EDA", "is_peak", "classes"],
                     sep=';')
    part_classes = df['classes'][:-1]
    part_peak = df['is_peak'][:-1]
    part_eda = [np.array(eval(elem), dtype=np.float32) for elem in df['EDA'].array]
    part_eda.pop()

    # Create a dictionary to represent the new row
    part_df = pd.DataFrame({
        'classes': part_classes.tolist(),
        'is_peak': part_peak.tolist(),
        'eda': part_eda
    })
    # Concatenate the part_df to the combined_df along rows
    combined_df = pd.concat([combined_df, part_df], ignore_index=True)
combined_df['is_peak'] = combined_df['is_peak'].astype(np.float32)

print("[Info] Dataframe combined")

X = tf.constant(combined_df['eda'].tolist())
X_peak = combined_df['is_peak'].to_numpy().reshape((X.shape[0], 1))

X = tf.concat([X, X_peak], axis=1).numpy()

if NUM_CLASSES == 2:
    Y = to_categorical(combined_df['classes'].replace({-1: 0, 1: 1}), num_classes=2)
else:
    Y = to_categorical(combined_df['classes'], num_classes=3)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
lol = 2
print("[Info] Dataset split")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", Y_train.shape)
print("y_test shape:", Y_test.shape)

model = get_basic_model()
model.fit(X_train, Y_train, epochs=6, batch_size=32)
test_loss, test_accuracy = model.evaluate(X_test, Y_test)

# =================================NEW CODE ====================================
# df_train = pd.read_csv(
#     "nn_dataset/P16.csv",
#     usecols=["EDA", "is_peak", "classes"],
#     sep=';')
#
# classes = df_train['classes']
# is_peak = df_train['is_peak']
#
# classes = classes.drop(classes.index[-1])
# is_peak = is_peak.drop(is_peak.index[-1])
#
# y = 2
#
# eda_features = [np.array(eval(elem), dtype=np.float32) for elem in df_train['EDA'].array]
# eda_features.pop()  # delete last element, that has different shape
#
# eda_tensor = tf.constant(eda_features)
#
# new_feature_array = is_peak.to_numpy().reshape((7500, 1))
# eda_tensor_with_new_feature = tf.concat([eda_tensor, new_feature_array], axis=1)
# y = 3
#
# model = get_basic_model()
# one_hot_classes = to_categorical(classes, num_classes=3)
# y = 2
# model.fit(eda_tensor_with_new_feature, one_hot_classes, epochs=100, batch_size=64)
