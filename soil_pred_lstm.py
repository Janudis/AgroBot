# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime

# Clear workspace (Not necessary in Python, but you can restart the kernel if needed)

# Import the CSV file
data_path = r'D:\NASA_Space_Apps\data\1.csv'
data = pd.read_csv(data_path, dtype={'time': str})  # Ensure 'time' is read as string

# Convert time to datetime
data['time'] = pd.to_datetime(data['time'])

# Sort the data by time
data = data.sort_values('time').reset_index(drop=True)

# Extract features and target
features = data[['latitude', 'longitude', 'clay_content', 'sand_content', 'silt_content', 'sm_aux']].values
features[:, 1] = features[:, 1] * 0.0001  # Adjust longitude

target = data['sm_tgt'].values

# Normalize the features and target
feature_scaler = MinMaxScaler()
features_norm = feature_scaler.fit_transform(features)

target_scaler = MinMaxScaler()
target_norm = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()

# Prepare sequences for LSTM
sequence_length = 30  # Use 30 days of data to predict the next day
num_features = features.shape[1]
num_samples = len(data) - sequence_length

X = []
Y = []

for i in range(num_samples):
    X.append(features_norm[i:i+sequence_length, :])
    Y.append(target_norm[i+sequence_length])

X = np.array(X)  # Shape: (num_samples, sequence_length, num_features)
Y = np.array(Y)  # Shape: (num_samples,)

# Split data into training and testing sets (80% train, 20% test)
train_size = int(0.8 * num_samples)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Define LSTM network architecture
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, num_features)),
    LSTM(50),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Set training options
epochs = 10
batch_size = 32

# Train the network
history = model.fit(
    X_train, Y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, Y_test),
    verbose=1
)

# Make predictions
YPred = model.predict(X_test).flatten()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(Y_test, YPred))
print(f'Test RMSE: {rmse:.6f}')

# Denormalize the predictions
YPred_denorm = target_scaler.inverse_transform(YPred.reshape(-1, 1)).flatten()
YTest_denorm = target_scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(data['time'][train_size + sequence_length:], YTest_denorm, label='Actual', color='blue')
plt.plot(data['time'][train_size + sequence_length:], YPred_denorm, label='Predicted', color='red')
plt.xlabel('Time')
plt.ylabel('Soil Moisture')
plt.title('Soil Moisture Prediction')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate and print evaluation metrics
mae = mean_absolute_error(YTest_denorm, YPred_denorm)
mape = np.mean(np.abs((YTest_denorm - YPred_denorm) / YTest_denorm)) * 100
r2 = r2_score(YTest_denorm, YPred_denorm)

print(f'Mean Absolute Error: {mae:.6f}')
print(f'Mean Absolute Percentage Error: {mape:.2f}%')
print(f'R-squared: {r2:.6f}')

# Plot feature importance
# Note: LSTM models are not inherently interpretable. Here, we'll use a simple proxy by computing
# the mean absolute difference of normalized features as in the MATLAB code.
feature_importance = np.mean(np.abs(np.diff(features_norm, axis=0)), axis=0)
plt.figure(figsize=(10, 6))
plt.bar(['Latitude', 'Longitude', 'Clay', 'Sand', 'Silt', 'SM Aux'], feature_importance)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save the LSTM model to a file
model_save_path = r'D:\NASA Space Apps 2024\SoilMoisturePredictorLSTM.h5'
model.save(model_save_path)
print(f'LSTM model saved to {model_save_path}')

# Generate C code for the LSTM model
# Python does not have a direct equivalent to MATLAB's codegen for C code generation.
# However, you can export the model to TensorFlow Lite for deployment on embedded systems.

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
tflite_model_path = r'D:\NASA Space Apps 2024\SoilMoisturePredictorLSTM.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
print(f'TensorFlow Lite model saved to {tflite_model_path}')

# If you specifically need C code, you might consider using tools like TensorFlow Lite Micro
# which allows embedding the model into C/C++ projects.
