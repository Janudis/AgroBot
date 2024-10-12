import os
import time
import google.generativeai as genai
import ee
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()
ee.Authenticate()
ee.Initialize(project='nsa-agroai')

# Configure Streamlit page settings
st.set_page_config(
    page_title="Chat with AgroAI!",
    page_icon=":brain:",  # Favicon emoji
    layout="centered",  # Page layout option
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def wait_for_files_active(files):
    """Waits for the given files to be active."""
    print("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    print("...all files ready")
    print()

def get_modis_ndvi(region, start_date, end_date):
    """Retrieve MODIS NDVI data from Earth Engine."""
    dataset = ee.ImageCollection('MODIS/006/MOD13A2') \
                .filterDate(start_date, end_date) \
                .filterBounds(region) \
                .select('NDVI')  # NDVI band
    
    ndvi_data = dataset.mean().reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=1000,  # Set the scale to 1 km resolution
        maxPixels=1e9
    ).getInfo()
    
    return ndvi_data

def get_smap_soil_moisture(region, start_date, end_date):
    """Retrieve SMAP Soil Moisture data from the SMAP SPL4SMGP/007 dataset."""
    # Select SMAP SPL4SMGP dataset and filter by date and region
    dataset = ee.ImageCollection("NASA/SMAP/SPL4SMGP/007") \
                .filterDate(start_date, end_date) \
                .filterBounds(region) \
                .select('sm_surface')

    # Check the first image for available bands (optional debug step)
    first_image = dataset.first()
    available_bands = first_image.bandNames().getInfo()
    print("Available SMAP Bands:", available_bands)  # Debug: List all available bands

    # The expected band for surface soil moisture is 'sm_surface'
    soil_moisture_band = 'sm_surface' if 'sm_surface' in available_bands else available_bands[0]  # Fallback to first available band

    # Reduce the image collection to get mean soil moisture for the time period
    soil_moisture_data = dataset.select(soil_moisture_band).mean().reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=9000,  # 9 km resolution for SMAP data
        maxPixels=1e9
    ).getInfo()

    return soil_moisture_data
def get_surface_temperature(region, start_date, end_date, temp_type='day'):
    """Retrieve surface temperature data from MODIS MOD11A1."""
    # Choose the band based on temp_type (day or night)
    if temp_type == 'day':
        temp_band = 'LST_Day_1km'
    elif temp_type == 'night':
        temp_band = 'LST_Night_1km'
    else:
        raise ValueError("Invalid temp_type. Use 'day' or 'night'.")

    # Load the MODIS MOD11A1 dataset and filter by date and region
    dataset = ee.ImageCollection("MODIS/061/MOD11A1") \
                .filterDate(start_date, end_date) \
                .filterBounds(region)

    # Check the first image for available bands (optional debug step)
    first_image = dataset.first()
    available_bands = first_image.bandNames().getInfo()
    print("Available MODIS Bands:", available_bands)  # Debug: List all available bands

    # Ensure the selected temperature band exists
    if temp_band not in available_bands:
        raise KeyError(f"{temp_band} is not available in the dataset.")

    # Reduce the image collection to get mean temperature for the time period
    # Convert from Kelvin to Celsius by subtracting 273.15
    surface_temperature_data = dataset.select(temp_band).mean().subtract(273.15).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=1000,  # 1 km resolution for MODIS
        maxPixels=1e9
    ).getInfo()

    return surface_temperature_data
def get_co_concentration(region, start_date, end_date):
    """Retrieve carbon monoxide concentration from Sentinel-5P NRTI CO dataset."""
    # Load the Sentinel-5P NRTI CO dataset and filter by date and region
    dataset = ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_CO") \
                .filterDate(start_date, end_date) \
                .filterBounds(region)

    # Check the first image for available bands (optional debug step)
    first_image = dataset.first()
    available_bands = first_image.bandNames().getInfo()
    print("Available Sentinel-5P CO Bands:", available_bands)  # Debug: List all available bands

    # Ensure the CO band exists
    co_band = 'CO_column_number_density'
    if co_band not in available_bands:
        raise KeyError(f"{co_band} is not available in the dataset.")

    # Reduce the image collection to get mean CO concentration for the time period
    co_data = dataset.select(co_band).mean().reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=1000,  # Adjust the scale based on the dataset resolution
        maxPixels=1e9
    ).getInfo()

    return co_data
def get_uv_aerosol_index(region, start_date, end_date):
    """Retrieve UV Aerosol Index from Sentinel-5P NRTI AER AI dataset."""
    # Load the Sentinel-5P NRTI AER AI dataset and filter by date and region
    dataset = ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_AER_AI") \
                .filterDate(start_date, end_date) \
                .filterBounds(region)

    # Check the first image for available bands (optional debug step)
    first_image = dataset.first()
    available_bands = first_image.bandNames().getInfo()
    print("Available Sentinel-5P AER AI Bands:", available_bands)  # Debug: List all available bands

    # Use the correct band name for the UV aerosol index
    ai_band = 'absorbing_aerosol_index'
    if ai_band not in available_bands:
        raise KeyError(f"{ai_band} is not available in the dataset.")

    # Reduce the image collection to get mean UV aerosol index for the time period
    uv_aerosol_index_data = dataset.select(ai_band).mean().reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=1000,  # Adjust the scale based on the dataset resolution
        maxPixels=1e9
    ).getInfo()

    return uv_aerosol_index_data
# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

# File management - uploading weather data as before
files = [
    upload_to_gemini("soil_moisture.csv", mime_type="text/csv"),
]
wait_for_files_active(files)

# Initialize chat session
chat_session = model.start_chat()
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[
        {
            "role": "user",
            "parts": [files[0]],
        },
        {
            "role": "model",
            "parts": [
                "This is weather data in CSV format. Here's a breakdown of the columns:\n\n<...continued description as before...>"
            ],
        },
    ])

# Display the chatbot's title on the page
st.title("🍃 AgroBot Chat Assistant")

# Display chat history
for message in st.session_state.chat_session.history:
    with st.chat_message(translate_role_for_streamlit(message.role)):
        st.markdown(message.parts[0].text)

# User input for MODIS/SMAP query
user_prompt = st.chat_input("Ask AgroBot...")

if user_prompt:
    # Add user's message to chat and display it
    st.chat_message("user").markdown(user_prompt)
    
    if "MODIS" in user_prompt or "NDVI" in user_prompt or "vegetation" in user_prompt:
        # Default region and date range for testing
        region = ee.Geometry.Point([-95.4107, 31.7662])  # e.g., Athens, Greece
        start_date = '2023-01-01'
        end_date = '2023-12-31'
        
        # Retrieve MODIS NDVI data
        ndvi_data = get_modis_ndvi(region, start_date, end_date)
        
        # Prepare response
        ndvi_response = f"Here is the average NDVI from MODIS for the selected region between {start_date} and {end_date}: {ndvi_data['NDVI']:.4f}."
        
        # Display MODIS NDVI data in the chatbot
        with st.chat_message("assistant"):
            st.markdown(ndvi_response)
    
    elif "surface soil moisture" in user_prompt or "SMAP" in user_prompt:
        # Default region and date range for testing
        region = ee.Geometry.Point([-95.4107, 31.7662])  # e.g., Athens, Greece
        start_date = '2023-01-01'
        end_date = '2023-12-31'
        
        # Retrieve SMAP soil moisture data
        soil_moisture_data = get_smap_soil_moisture(region, start_date, end_date)
        
        # Prepare response
        soil_moisture_response = f"Here is the average surface soil moisture from SMAP for the selected region between {start_date} and {end_date}: {soil_moisture_data['sm_surface']:.4f} cm³/cm³."
        
        # Display SMAP soil moisture data in the chatbot
        with st.chat_message("assistant"):
            st.markdown(soil_moisture_response)
    elif "surface temperature" in user_prompt or "LST" in user_prompt:
        # Default region and date range for testing
        region = ee.Geometry.Point([-95.4107, 31.7662])  # e.g., Athens, Greece
        start_date = '2023-01-01'
        end_date = '2023-12-31'
        
        # Retrieve SMAP soil moisture data
        surface_temp_day = get_surface_temperature(region, start_date, end_date, temp_type='day')
        
        # Prepare response
        surface_temp_response = f"Daytime surface temperature in Texas is {round(surface_temp_day['LST_Day_1km']*0.02 - 273.15,2)} celsius"
        
        # Display SMAP soil moisture data in the chatbot
        with st.chat_message("assistant"):
            st.markdown(surface_temp_response) 
    elif "carbon monoxide" in user_prompt or "Sentinel 5P" in user_prompt or "CO" in user_prompt:
        region = ee.Geometry.Point([-95.4107, 31.7662])  # Coordinates for Athens, Greece
        start_date = '2023-01-01'
        end_date = '2023-12-31'
        # Get carbon monoxide concentration
        co_concentration = get_co_concentration(region, start_date, end_date)
        co_concentration_response = f"Carbon monoxide concentration (mol/m²): {round(co_concentration['CO_column_number_density'])}"
        
        # Display SMAP soil moisture data in the chatbot
        with st.chat_message("assistant"):
            st.markdown(co_concentration_response) 
    elif "aerosol" in user_prompt or "Sentinel 5P" in user_prompt:
        region = ee.Geometry.Point([-95.4107, 31.7662])  # Coordinates for Athens, Greece
        start_date = '2023-01-01'
        end_date = '2023-12-31'
        # Get carbon monoxide concentration
        uv_aerosol_index = get_uv_aerosol_index(region, start_date, end_date)
        uv_aerosol_index_response = f"UV Aerosol Index (AER_AI_340_380): {uv_aerosol_index}"
        
        # Display SMAP soil moisture data in the chatbot
        with st.chat_message("assistant"):
            st.markdown(uv_aerosol_index_response) 
    else:
        # Send user's message to Gemini-Pro and get the response
        gemini_response = st.session_state.chat_session.send_message(user_prompt)
        
        # Display Gemini-Pro's response
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)
