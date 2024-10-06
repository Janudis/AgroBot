from soil_pred_lstm import *
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from datetime import timedelta

# Load the trained model
model_save_path = r'D:\NASA_Space_Apps\results\SoilMoisturePredictorLSTM.h5'
model = load_model(model_save_path)
print('Trained model loaded.')

# Load the scalers
scalers_path = r'D:\NASA_Space_Apps\results\scalers.pkl'
with open(scalers_path, 'rb') as f:
    scalers = pickle.load(f)
print('Scalers loaded.')

# Extract individual scalers
geo_scaler = scalers['geo_scaler']
soil_scaler = scalers['soil_scaler']
moisture_scaler = scalers['moisture_scaler']
temporal_scaler = scalers['temporal_scaler']
target_scaler = scalers['target_scaler']

# Define sequence_length (should match the one used during training)
sequence_length = 30  # Use 30 days of data to predict the next day

# Load the data
data_path = r'D:\NASA_Space_Apps\data\updated_data.csv'
data = pd.read_csv(data_path, dtype={'time': str})  # Ensure 'time' is read as string

# Convert 'time' to datetime and sort the data
data['time'] = pd.to_datetime(data['time'])
data = data.sort_values('time').reset_index(drop=True)

# Incorporate temporal features
data['month'] = data['time'].dt.month
data['day_of_year'] = data['time'].dt.dayofyear

# Define the predict_soil_moisture function
def predict_soil_moisture(location_lat, location_lon, start_date, num_days_to_predict, model, scalers):
    """
    Predict soil moisture for a specific location and number of future days.
    """
    # Extract scalers
    geo_scaler = scalers['geo_scaler']
    soil_scaler = scalers['soil_scaler']
    moisture_scaler = scalers['moisture_scaler']
    temporal_scaler = scalers['temporal_scaler']
    target_scaler = scalers['target_scaler']

    # Filter data for the specific location
    location_data = data[(data['latitude'] == location_lat) & (data['longitude'] == location_lon)].copy()
    location_data = location_data.sort_values('time').reset_index(drop=True)
    
    if len(location_data) < sequence_length:
        print("Not enough data for the specified location to create input sequences.")
        return None
    
    # Get the last sequence_length days of data
    last_sequence = location_data.iloc[-sequence_length:].copy()
    
    # Prepare features for the last_sequence
    geo_features = last_sequence[['latitude', 'longitude']].values
    soil_features = last_sequence[['clay_content', 'sand_content', 'silt_content']].values
    moisture_features = last_sequence[['sm_aux']].values
    temporal_features = last_sequence[['month', 'day_of_year']].values
    
    # Scale features using the same scalers
    geo_scaled = geo_scaler.transform(geo_features)
    soil_scaled = soil_scaler.transform(soil_features)
    moisture_scaled = moisture_scaler.transform(moisture_features)
    temporal_scaled = temporal_scaler.transform(temporal_features)
    
    features_scaled = np.hstack((geo_scaled, soil_scaled, moisture_scaled, temporal_scaled))
    
    # Initialize list to store predictions
    predictions = []
    dates = []
    
    # Start date for predictions
    current_date = pd.to_datetime(start_date)
    
    # Loop to predict future days
    for i in range(num_days_to_predict):
        # Prepare input data for the model
        input_sequence = features_scaled[-sequence_length:]  # Get the last sequence_length entries
        input_sequence = np.expand_dims(input_sequence, axis=0)  # Shape: (1, sequence_length, num_features)
        
        # Predict the next day's soil moisture
        pred_scaled = model.predict(input_sequence)
        pred_denorm = target_scaler.inverse_transform(pred_scaled).flatten()[0]
        
        # Store the prediction
        predictions.append(pred_denorm)
        dates.append(current_date)
        
        # Prepare features for the next day
        # For features that change daily, we need to update them
        next_day = current_date + timedelta(days=1)
        next_month = next_day.month
        next_day_of_year = next_day.dayofyear
        
        # Scale temporal features
        temporal_next = np.array([[next_month, next_day_of_year]])
        temporal_next_scaled = temporal_scaler.transform(temporal_next)
        
        # For sm_aux, if future values are unknown, we can use the last known value or estimate it
        # Here, we use the last known value
        sm_aux_last = moisture_scaled[-1]
        
        # Prepare the next day's feature vector
        next_feature = np.hstack((
            geo_scaled[-1],           # Latitude and Longitude (unchanged)
            soil_scaled[-1],          # Soil composition (unchanged)
            sm_aux_last,              # sm_aux (using last known value)
            temporal_next_scaled[0]   # Scaled temporal features
        ))
        
        # Append the next_feature to features_scaled for the next iteration
        features_scaled = np.vstack((features_scaled, next_feature))
        
        # Move to the next day
        current_date = next_day
    
    # Create DataFrame for the predictions
    predictions_df = pd.DataFrame({
        'date': dates,
        'latitude': location_lat,
        'longitude': location_lon,
        'predicted_soil_moisture': predictions
    })
    
    return predictions_df

# Example usage:
# Predict soil moisture for a specific location for the next 7 days
location_lat = 54.875  # Replace with desired latitude
location_lon = 9.375   # Replace with desired longitude
# location_lat = 32.7767  # Latitude for Dallas, Texas
# location_lon = -96.7970  # Longitude for Dallas, Texas
# start_date = data['time'].max() + pd.Timedelta(days=1)  # Start prediction from the day after the last date in data
start_date = pd.to_datetime('2024-10-01')  # Start prediction from October 1, 2024
num_days_to_predict = 7

predictions_df = predict_soil_moisture(location_lat, location_lon, start_date, num_days_to_predict, model, scalers)

if predictions_df is not None:
    # Output predictions to CSV
    output_csv_path = r'D:\NASA_Space_Apps\results\predictions.csv'
    predictions_df.to_csv(output_csv_path, index=False)
    print(f'Predictions saved to {output_csv_path}')
    print(predictions_df)
else:
    print("Prediction was not successful.")
