# Import necessary libraries
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import shap
import datetime
import os
import pickle
# Suppress TensorFlow warnings for cleaner output
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# # Set random seed for reproducibility
# np.random.seed(42)
# tf.random.set_seed(42)

def main():
    # 1. Data Loading and Preprocessing
    # Define the path to the CSV file
    data_path = r'D:\NASA_Space_Apps\data\updated_data.csv'

    # Load the data
    data = pd.read_csv(data_path, dtype={'time': str})  # Ensure 'time' is read as string

    # Convert 'time' to datetime and sort the data
    data['time'] = pd.to_datetime(data['time'])
    data = data.sort_values('time').reset_index(drop=True)

    # Handle duplicates and missing values
    # Assuming that duplicates are defined by the same 'time', 'latitude', 'longitude'
    data = data.drop_duplicates(subset=['time', 'latitude', 'longitude'])
    data = data.dropna()

    # Extract features and target
    features = data[['latitude', 'longitude', 'clay_content', 'sand_content', 'silt_content', 'sm_aux']].values
    target = data['sm_tgt'].values

    # Analyze feature distribution
    print("Feature statistics before scaling:")
    print(pd.DataFrame(features, columns=['Latitude', 'Longitude', 'Clay', 'Sand', 'Silt', 'SM Aux']).describe())

    # Group features based on their nature for appropriate scaling
    geo_features = features[:, :2]       # Latitude and Longitude
    soil_features = features[:, 2:5]     # Clay, Sand, Silt Content
    moisture_features = features[:, 5:]  # sm_aux

    # Initialize scalers
    geo_scaler = MinMaxScaler()
    soil_scaler = MinMaxScaler()
    moisture_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Scale geographical features (Latitude and Longitude)
    geo_scaled = geo_scaler.fit_transform(geo_features)

    # Scale soil composition features (Clay, Sand, Silt)
    soil_scaled = soil_scaler.fit_transform(soil_features)

    # Scale sm_aux
    moisture_scaled = moisture_scaler.fit_transform(moisture_features)

    # Combine all scaled features
    features_scaled = np.hstack((geo_scaled, soil_scaled, moisture_scaled))

    # Scale the target variable
    target_scaled = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()

    # Optional: Incorporate temporal features
    data['month'] = data['time'].dt.month
    data['day_of_year'] = data['time'].dt.dayofyear

    # Extract temporal features
    temporal_features = data[['month', 'day_of_year']].values

    # Scale temporal features
    temporal_scaler = MinMaxScaler()
    temporal_scaled = temporal_scaler.fit_transform(temporal_features)

    # Append temporal features to scaled features
    features_final = np.hstack((features_scaled, temporal_scaled))

    # Update the number of features
    num_features = features_final.shape[1]

    # Prepare sequences for LSTM
    sequence_length = 30  # Use 30 days of data to predict the next day
    num_samples = len(data) - sequence_length

    X = []
    Y = []

    for i in range(num_samples):
        X.append(features_final[i:i+sequence_length, :])
        Y.append(target_scaled[i+sequence_length])

    X = np.array(X)  # Shape: (num_samples, sequence_length, num_features)
    Y = np.array(Y)  # Shape: (num_samples,)

    # Split data into training and testing sets (80% train, 20% test)
    train_size = int(0.8 * num_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    print(f"Total samples: {num_samples}")
    print(f"Training samples: {train_size}")
    print(f"Testing samples: {num_samples - train_size}")

    # 2. Model Building and Training

    # Define LSTM network architecture with Dropout for regularization
    model = Sequential([
        Input(shape=(sequence_length, num_features)),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Define early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    # checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
    # Set training options
    epochs = 50  # Increased epochs for better learning
    batch_size = 32  # Adjusted batch size
    # Train the network
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, Y_test),
        callbacks=[early_stop],
        verbose=1
    )

    # 3. Model Evaluation
    # Make predictions on the test set
    YPred = model.predict(X_test).flatten()

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(Y_test, YPred))
    mae = mean_absolute_error(Y_test, YPred)
    r2 = r2_score(Y_test, YPred)

    print(f'\nTest RMSE: {rmse:.6f}')
    print(f'Test MAE: {mae:.6f}')
    print(f'Test R-squared: {r2:.6f}')

    # Denormalize the predictions and actual values
    YPred_denorm = target_scaler.inverse_transform(YPred.reshape(-1, 1)).flatten()
    YTest_denorm = target_scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()

    # Plot Actual vs. Predicted Soil Moisture
    plt.figure(figsize=(15, 7))
    plt.plot(data['time'][train_size + sequence_length:], YTest_denorm, label='Actual', color='blue')
    plt.plot(data['time'][train_size + sequence_length:], YPred_denorm, label='Predicted', color='red')
    plt.xlabel('Time')
    plt.ylabel('Soil Moisture')
    plt.title('Soil Moisture Prediction: Actual vs. Predicted')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot Training and Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss', color='green')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Residual Analysis: Plot Residuals
    residuals = YTest_denorm - YPred_denorm

    plt.figure(figsize=(15, 5))
    plt.scatter(YTest_denorm, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Actual Soil Moisture')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Actual Soil Moisture')
    plt.tight_layout()
    plt.show()

    # Histogram of Residuals
    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=50, color='purple', edgecolor='k')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.tight_layout()
    plt.show()

    # Feature Importance Using SHAP
    # Note: SHAP can be computationally intensive for LSTM models.
    # We'll use a subset of the data for demonstration purposes.
    # # Initialize SHAP DeepExplainer
    # # Use a small subset to speed up computation
    # shap_sample = X_train[:100]
    # explainer = shap.DeepExplainer(model, shap_sample)
    # shap_values = explainer.shap_values(X_test[:100])
    # # Since shap_values is a list for each output, we take the first element
    # shap_values = shap_values[0]
    # # Aggregate SHAP values across the sequence
    # # For visualization, we'll average the SHAP values across the sequence length
    # shap_values_agg = np.mean(shap_values, axis=1)  # Shape: (100, num_features)
    # # Define feature names including temporal features
    # feature_names = ['Latitude', 'Longitude', 'Clay', 'Sand', 'Silt', 'SM Aux', 'Month', 'Day of Year']
    # # Plot SHAP summary
    # plt.figure(figsize=(12, 8))
    # shap.summary_plot(shap_values_agg, features_final[train_size:train_size+100, :], feature_names=feature_names, show=False)
    # plt.tight_layout()
    # plt.show()

    # 4. Model Saving and Deployment

    # Define paths for saving the model
    model_save_path = r'D:\NASA_Space_Apps\results\SoilMoisturePredictorLSTM.h5'
    tflite_model_path = r'D:\NASA_Space_Apps\results\SoilMoisturePredictorLSTM.tflite'

    # Save the Keras model
    model.save(model_save_path)
    print(f'\nLSTM model saved to {model_save_path}')

    # Convert the model to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Set converter options to handle resource variables and select TF ops
    converter.experimental_enable_resource_variables = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,    # Enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS       # Enable select TensorFlow ops.
    ]
    converter._experimental_lower_tensor_list_ops = False  # Disable tensor list lowering

    tflite_model = converter.convert()

    # Save the TensorFlow Lite model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f'TensorFlow Lite model saved to {tflite_model_path}')

    scalers = {
    'geo_scaler': geo_scaler,
    'soil_scaler': soil_scaler,
    'moisture_scaler': moisture_scaler,
    'temporal_scaler': temporal_scaler,
    'target_scaler': target_scaler
    }
    scalers_path = r'D:\NASA_Space_Apps\results\scalers.pkl'
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f'Scalers saved to {scalers_path}')

# Ensure this code runs only when executing the script directly
if __name__ == "__main__":
    main()