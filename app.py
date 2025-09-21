# app.py - PREMIUM UI REFINED (Compact & Professional)
# ================================================
# This Streamlit app calculates actuarial insurance premiums using a 
# pre-trained Gradient Boosting Machine (GBM) model. Users provide risk
# factors such as driver age, vehicle age, and garaging location, and
# the app predicts premiums in both USD and UGX.
# ================================================

import streamlit as st
import pandas as pd
import joblib

# ========== GLOBAL CONSTANTS & CONFIG ==========
EXCHANGE_RATE = 3700  # Conversion rate USD → UGX for demonstration purposes.
# In production, consider fetching real-time FX rates from an API.

# Streamlit page configuration
st.set_page_config(
    page_title="Actuarial Premium Calculator",
    page_icon="📊",
    layout="wide",  # Wider layout for better info card display
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS STYLING ==========
# Styling applied globally to improve UX/UI (info cards, headers, buttons, etc.)
APP_CSS = """
<style>
    /* Overall page layout */
    .block-container {
        max-width: 1050px;
        margin: auto;
        padding-top: 2rem;
    }
    /* Header style */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
        color: #1a202c;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    /* Info cards to display input parameters */
    .info-card {
        background: #f9f9fd;
        border-radius: 14px;
        padding: 1rem;
        border: 1px solid #e5e7eb;
        text-align: center;
        min-height: 110px;
    }
    .info-label { font-size: 1.2rem; color: #4b3b82; margin-bottom: .4rem; font-weight: 650; }
    .info-value { font-size: 2rem; font-weight: 700; color: #1a202c; }

    /* Prediction results styling */
    .result-title { font-size: 1.2rem; font-weight: 600; color: #374151; }
    .result-highlight { font-size: 2.6rem; font-weight: 800; margin-top: .5rem; color: #7c3aed; }
    .ugx-result { font-size: 2.2rem; font-weight: 700; margin-top: .5rem; color: #7c3aed; }

    /* Sidebar input labels */
    .stNumberInput label, .stSelectbox label { font-size: 1.1rem !important; font-weight: 600 !important; color: #91abe3 !important; }

    /* Buttons */
    div.stButton > button {
        width: 100%;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        padding: .7rem 1rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        transition: background 0.3s ease, transform 0.2s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, #764ba2, #667eea);
    }

    /* Remove default footer and toolbar */
    footer, .stApp [data-testid="stToolbar"] { display: none !important; }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

# ========== CORE FUNCTIONS ==========

@st.cache_resource
def load_model(model_path='gbm_auto_model.pkl'):
    """
    Load the pre-trained GBM model from file.
    Uses caching to prevent reloading on every interaction.
    Args:
        model_path (str): Path to the pickled model file.
    Returns:
        model: Loaded GBM model ready for predictions.
    """
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"❌ Model file not found. Ensure '{model_path}' is in the app directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ An error occurred loading the model: {e}")
        st.stop()

def format_input_data(age, vehicle_age, location):
    """
    Converts user inputs into a DataFrame matching the model's expected format.
    Args:
        age (int): Driver's age
        vehicle_age (int): Vehicle's age in years
        location (str): Garaging location ("Urban", "Suburban", "Rural")
    Returns:
        pd.DataFrame: Single-row DataFrame ready for model.predict()
    """
    input_dict = {
        'age': [age],
        'vehicle_age': [vehicle_age],
        'location_Rural': [1 if location == "Rural" else 0],
        'location_Suburban': [1 if location == "Suburban" else 0],
        'location_Urban': [1 if location == "Urban" else 0]
    }
    # Ensure columns follow the same order as model training
    model_feature_order = ['age', 'vehicle_age', 'location_Rural', 'location_Suburban', 'location_Urban']
    input_df = pd.DataFrame(input_dict)
    return input_df[model_feature_order]

# ========== MAIN APPLICATION ==========
def main():
    """
    Main Streamlit app function.
    Handles UI layout, input collection, prediction, and results display.
    """
    # Load the pre-trained model
    model = load_model()

    # Define layout: sidebar (inputs) vs main content (display)
    sidebar_col, main_content_col = st.columns([1, 2.8])

    # ----- SIDEBAR: User Inputs -----
    with sidebar_col:
        st.markdown("### Risk Assessment Parameters")
        age = st.number_input(
            "Driver Age", min_value=18, max_value=80, value=40, step=1,
            help="Age of the primary policyholder (18-80 years)", key="age_input"
        )
        vehicle_age = st.number_input(
            "Vehicle Age (years)", min_value=1, max_value=20, value=5, step=1,
            help="Age of the vehicle being insured (1-20 years)", key="vehicle_input"
        )
        location = st.selectbox(
            "Garaging Location", options=["Urban", "Suburban", "Rural"],
            help="Where the vehicle is primarily parked overnight", key="location_input"
        )
        calculate_btn = st.button("**Calculate Premium**", type="primary", use_container_width=True)

    # ----- MAIN CONTENT: Display & Prediction -----
    with main_content_col:
        # Header
        st.markdown('<h1 class="main-header">Actuarial Premium Calculator</h1>', unsafe_allow_html=True)
        st.markdown("**Powered by Machine Learning** - Estimates insurance premiums using GBM model.")
        st.markdown("---")

        # Display input summary in info cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="info-card"><div class="info-label">👤 Driver Age</div><div class="info-value">{age} yrs</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="info-card"><div class="info-label">🚗 Vehicle Age</div><div class="info-value">{vehicle_age} yrs</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="info-card"><div class="info-label">📌 Location</div><div class="info-value">{location}</div></div>', unsafe_allow_html=True)

        # Prediction & result display
        if calculate_btn:
            with st.spinner("🔍 Analyzing risk profile..."):
                # Format input data for the model
                input_df = format_input_data(age, vehicle_age, location)
                # Predict premium in USD
                prediction_usd = model.predict(input_df)[0]
                # Convert to UGX
                prediction_ugx = prediction_usd * EXCHANGE_RATE

            # Display prediction side-by-side
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<div class="result-title">USD Premium</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-highlight">${prediction_usd:,.2f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with res_col2:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<div class="result-title">UGX Premium</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ugx-result">UGX {prediction_ugx:,.0f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # ----- FOOTER -----
    st.markdown("---")
    st.caption("© 2025 - Group 9 | BSAS Data Analysis III")  # Footer for attribution

# ========== RUN APP ==========
if __name__ == "__main__":
    main()
