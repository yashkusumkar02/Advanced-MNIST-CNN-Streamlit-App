import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import numpy as np
import pandas as pd

print("--- Starting Model Training with ADVANCED CNN ---")

# --- 1. Load Data from CSV ---
try:
    # Assuming 'label' is the first column in both CSVs
    train_df = pd.read_csv('mnist_train.csv')
    test_df = pd.read_csv('mnist_test.csv')
except FileNotFoundError:
    print("❌ Error: One or both MNIST CSV files not found. Please ensure 'mnist_train.csv' and 'mnist_test.csv' are in the same directory.")
    exit()

# Separate labels (Y) from pixel data (X)
y_train = train_df['label'].values
x_train = train_df.drop('label', axis=1).values

y_test = test_df['label'].values
x_test = test_df.drop('label', axis=1).values

# --- 2. Preprocess Data ---
N_train = x_train.shape[0]
N_test = x_test.shape[0]

# Reshape pixel data from (N, 784) to (N, 28, 28, 1) and normalize
x_train = x_train.reshape(N_train, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(N_test, 28, 28, 1).astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

print(f"Train data shape: {x_train.shape}, Test data shape: {x_test.shape}")

# --- 3. Build the ADVANCED CNN Model ---
model = Sequential([
    # Block 1: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> Pool
    Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25), # Regularization

    # Block 2: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> Pool
    Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25), # Regularization

    # Classification Head
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5), # Regularization
    Dense(10, activation='softmax') # Output layer: 10 classes (0-9)
])

# --- 4. Compile and Train ---
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

print("\nStarting model training...")
# Train the model 
model.fit(x_train, y_train, 
          epochs=10, # Increased epochs for better training of the deeper network
          batch_size=64, 
          validation_data=(x_test, y_test))

# --- 5. Save the Model ---
MODEL_PATH = 'mnist_cnn_model.h5'
model.save(MODEL_PATH)
print(f"\n✅ Model training complete and saved as: {MODEL_PATH}")
