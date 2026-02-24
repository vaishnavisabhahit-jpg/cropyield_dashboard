import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(page_title="Crop Yield Prediction", layout="wide")

st.title("🌾 Crop Yield Prediction Dashboard")
st.markdown("Predict crop yield based on soil & environmental conditions")

# -----------------------------------
# Load Model & Column Names FIRST
# -----------------------------------
model = pickle.load(open("crop_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

# -----------------------------------
# Sidebar Inputs
# -----------------------------------
st.sidebar.header("📝 Input Parameters")

states = [
    "Maharashtra",
    "Karnataka",
    "Tamil Nadu",
    "Punjab",
    "Uttar Pradesh",
    "Bihar"
]

crops = [
    "Rice",
    "Wheat",
    "Maize",
    "Cotton",
    "Sugarcane",
    "Barley"
]

selected_state = st.sidebar.selectbox("Select State", states)
selected_crop = st.sidebar.selectbox("Select Crop", crops)

temperature = st.sidebar.number_input("Temperature", 0.0, 50.0, 25.0)
rainfall = st.sidebar.number_input("Rainfall", 0.0, 500.0, 100.0)
humidity = st.sidebar.number_input("Humidity", 0.0, 100.0, 50.0)

# -----------------------------------
# Prepare Input Data (AFTER loading model_columns)
# -----------------------------------

# Create dataframe with exact training columns
input_df = pd.DataFrame(columns=model_columns)

# Add one row of zeros
input_df.loc[0] = 0

# Fill numerical columns ONLY if they exist
if "temperature" in input_df.columns:
    input_df.at[0, "temperature"] = temperature

if "rainfall" in input_df.columns:
    input_df.at[0, "rainfall"] = rainfall

if "humidity" in input_df.columns:
    input_df.at[0, "humidity"] = humidity

# One-hot encode state
state_column = f"State_{selected_state}"
if state_column in input_df.columns:
    input_df.at[0, state_column] = 1

# One-hot encode crop
crop_column = f"Crop_{selected_crop}"
if crop_column in input_df.columns:
    input_df.at[0, crop_column] = 1

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("🌱 Predict Yield"):

    prediction = model.predict(input_df)[0]

    st.subheader("📊 Prediction Result")
    st.success(f"🌾 Estimated Yield: {round(prediction, 2)} tons/hectare")

# -----------------------------------
# Show Selected Inputs
# -----------------------------------
st.subheader("📋 Selected Inputs")
st.write(f"State: {selected_state}")
st.write(f"Crop: {selected_crop}")
st.write(f"Temperature: {temperature}")
st.write(f"Rainfall: {rainfall}")
st.write(f"Humidity: {humidity}")