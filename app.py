import streamlit as st
import pandas as pd
import pickle

# Load the trained pipeline (model + preprocessing steps)
@st.cache_resource
def load_pipeline():
    with open("RidgeModel.pkl", "rb") as f:
        return pickle.load(f)

# Load unique dropdown values for selectboxes
@st.cache_resource
def load_dropdowns():
    with open("dropdown_values.pkl", "rb") as f:
        return pickle.load(f)

# Load model and dropdown values
pipeline = load_pipeline()
dropdown_values = load_dropdowns()

# Extract values for selectboxes
locations = dropdown_values["location"]
sqft_options = dropdown_values["total_sqft"]
bath_options = dropdown_values["bath"]
bhk_options = dropdown_values["bhk"]

# Sidebar with model description
st.sidebar.title("📘 Model Overview")
st.sidebar.markdown("""
### 🧠 Ridge Regression Model
This application uses a **Ridge Regression** model trained on real estate data from **Bangalore**.

- 🔍 **Features** used:
  - Location
  - Total Square Feet
  - Number of Bathrooms
  - BHK (Bedrooms, Hall, Kitchen)
  
- 📊 **Model Type**: Regularized Linear Regression (Ridge)
- 🏷️ **Output**: Predicted property price (in lakhs)

---

📎 *Note*: The values shown in the dropdowns are based on the training dataset for consistency and improved accuracy.
""")

# Main app title
st.title("🏠 Bangluru House Price Prediction")

# Location selection
location = st.selectbox("Select Location", locations)

# Total square feet selection
sqft = st.selectbox("Select Total Square Feet", sqft_options)

# Number of bathrooms selection
bath = st.selectbox("Select Number of Bathrooms", bath_options)

# Number of BHK selection
bhk = st.selectbox("Select Number of BHK", bhk_options)

# Prediction button
if st.button("Price Predict"):
    # Create input DataFrame for prediction
    input_df = pd.DataFrame([{
        "location": location,
        "total_sqft": sqft,
        "bath": bath,
        "bhk": bhk
    }])

    try:
        # Predict the price using the pipeline
        prediction = pipeline.predict(input_df)[0]
        st.success(f"🏷️ Estimated Price: ₹ {prediction:,.2f} lakhs")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
