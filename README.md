
# AgroBot

AgroBot is a chatbot assistant that can provide accurate agricultural satellite and other relevant data to the user, only by recieving a simple sentence as the input. The main of AgroBot is that any farmer can use this tool with or without technical or very high level scientific knowledge, resulting in easier process of making crucial decisions or predictions for their crops easier 


## Demo

https://agrobot.streamlit.app/


## Authors
The main authors of the code of AgroBot are:
- [Dimitris Ellinoudis](https://www.github.com/Janudis)
- [Alexandros Mylonas](https://www.github.com/almylonas)
- [Georgios Vellios](https://www.github.com/Niel518)\
\
For the scientific research that made AgroBot a reality contributed:
- [Vasilis Sakellaridis](https://github.com/moskovsakel)
- Melina Mavidou

## Satellite Data
AgroBot is able to recieve data from satellites using the Google Earth Engine API. GEE offers a variety of satellite datasets to be used in a variety of projects. We used the following satellite data:
| Satellite  | Instrument | Type of Data | Frequency of available data |
| ------------- | ------------- | ------------ | -------------
| MODIS  | NDVI  | Vegetation | 16-day (average)
| MODIS  | LST  | Surface temperature | daily
| SMAP  | -  | Soil moisture | 3-hourly
| Sentinel-5P  | -  | Carbon monoxide (CO) concentration | daily
| Sentinel-5P  | -  | Aerosol concentration| daily

In the future, we plan to increase the number of satellite datasets to provide a larger variety of information to the farmers.
## Datasets

AgroBot is able to read given data from .csv files. Generally, our team has searched and acquired data for soil moisture, air quality, floods, weather, surface temperature, droughts and vegetation growth. In this repository only the soil moisture dataset is provided, with data starting from 1/1/2002 to 22/5/2019. The dataset can be found in the ```chatbot``` folder under the name
```soil_moisture.csv```
## Predictions
AgroBot is also able to provide predictions to the user, based on an AI model we developed, designed to predict soil moisture levels using a Long Short-Term Memory (LSTM) neural network. The process began with data acquisition and preprocessing, where we gathered soil moisture data, geographical coordinates (latitude and longitude), soil composition metrics (clay, sand, silt content), and auxiliary soil moisture data (sm_aux) from various NASA datasets.

Our model is an LSTM neural network. We structured the network with two LSTM layers and incorporated dropout layers to prevent overfitting. The model was trained using the Adam optimizer and mean squared error as the loss function. We implemented early stopping to halt training when the validation loss ceased improving.

Then, we validated our model by generating predictions for specific locations and dates, outputting the results to CSV files for easy interpretation and further use and analysis. Finally, AgroBot can read these .csv files and provide predictions to the user.

The LSTM of AgroBot was developed in MATLAB.
## Installation

To clone the repository to your computer usen the following command line:

```bash
  git clone https://github.com/Janudis/AgroBot
```
Then, if you want to run the web app locally, then you have to install all the packages that are used in the web app (see main.py) and then run the following:
```bash
  cd AgroBot/chatbot
  streamlit run main.py
```
## Environment Variables

To run this project locally, you will need to add the following environment variables:

`GOOGLE_API_KEY`

You can acquire a google gemini api key at https://aistudio.google.com/app/apikey
## Datasets

AgroBot is able to read given data from .csv files. Generally, our team has searched and acquired data for soil moisture, air quality, floods, weather, surface temperature, droughts and vegetation growth. In this repository only the soil moisture dataset is provided, with data starting from 1/1/2002 to 22/5/2019. The dataset can be found in the ```chatbot``` folder under the name
```soil_moisture.csv  ```
