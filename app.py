# app.py
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="🌾 Crop Yield Predictor",
    page_icon="🌱",
    layout="wide"
)

st.title("🌾 Crop Yield Prediction Dashboard")
st.subheader("🌱 Predict crop yield based on soil and environmental conditions")

# ----------------------------
# Load model and columns
# ----------------------------
with open("crop_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# ----------------------------
# Sidebar: User inputs
# ----------------------------
st.sidebar.header("Input Parameters")

def user_input_features():
    data = {}
    for col in model_columns:
        # Slider for numeric inputs; adjust range if needed
        data[col] = st.sidebar.slider(col, 0, 100, 50)
    return pd.DataFrame([data])

input_df = user_input_features()

st.subheader("📝 User Input Summary")
st.dataframe(input_df)

# Display first two metrics dynamically (avoid KeyError)
st.metric(f"{model_columns[0]}", input_df[model_columns[0]][0])
if len(model_columns) > 1:
    st.metric(f"{model_columns[1]}", input_df[model_columns[1]][0])

# ----------------------------
# Make prediction
# ----------------------------
prediction = model.predict(input_df)

# ----------------------------
# Tabs for Prediction & Feature Importance
# ----------------------------
tab1, tab2 = st.tabs(["Prediction 🌟", "Feature Importance 📊"])

with tab1:
    st.subheader("Predicted Crop Yield (tons/hectare)")
    st.success(f"🌱 {prediction[0]:.2f} tons/hectare")

with tab2:
    st.subheader("Feature Importance")
    importance = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": model_columns, "Importance": importance})
    fi_df = fi_df.sort_values("Importance", ascending=False)
    
    fig = px.bar(
        fi_df, x="Feature", y="Importance",
        color="Importance", color_continuous_scale="Viridis",
        title="🌟 Feature Importance of Input Parameters"
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Footer / tip
# ----------------------------
st.markdown(
    "<h5 style='color:green;'>💡 Tip: Adjust the sliders in the sidebar to see live prediction changes!</h5>",
    unsafe_allow_html=True
)