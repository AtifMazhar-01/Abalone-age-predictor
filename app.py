import os

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()


# ENDPOINT_URL = os.getenv("AZURE_ENDPOINT_URL")
# ENDPOINT_KEY = os.getenv("AZURE_ENDPOINT_KEY")

ENDPOINT_URL = st.secrets["AZURE_ENDPOINT_URL"]
ENDPOINT_KEY = st.secrets["AZURE_ENDPOINT_KEY"]


st.set_page_config(
    page_title="Abalone Age Predictor",
    page_icon="🐚",
    layout="centered"
)


st.title("🐚 Abalone Age Predictor")
st.write("Enter the abalone measurements to predict its number of rings.")


# -------------------------
# Input fields
# -------------------------

sex = st.selectbox(
    "Sex",
    ["M", "F", "I"]
)


col1, col2 = st.columns(2)

with col1:
    length = st.number_input(
        "Length",
        value=0.455,
        format="%.4f"
    )

    height = st.number_input(
        "Height",
        value=0.095,
        format="%.4f"
    )

    shucked_weight = st.number_input(
        "Shucked Weight",
        value=0.2245,
        format="%.4f"
    )

    shell_weight = st.number_input(
        "Shell Weight",
        value=0.1500,
        format="%.4f"
    )


with col2:
    diameter = st.number_input(
        "Diameter",
        value=0.365,
        format="%.4f"
    )

    whole_weight = st.number_input(
        "Whole Weight",
        value=0.514,
        format="%.4f"
    )

    viscera_weight = st.number_input(
        "Viscera Weight",
        value=0.101,
        format="%.4f"
    )


# -------------------------
# Prediction
# -------------------------

if st.button("🔮 Predict", use_container_width=True):

    if not ENDPOINT_KEY:
        st.error("Azure ML endpoint key is missing.")
        st.stop()

    data = {
        "input_data": {
            "columns": [
                "Sex",
                "Length",
                "Diameter",
                "Height",
                "Whole_weight",
                "Shucked_weight",
                "Viscera_weight",
                "Shell_weight"
            ],
            "data": [
                [
                    sex,
                    length,
                    diameter,
                    height,
                    whole_weight,
                    shucked_weight,
                    viscera_weight,
                    shell_weight
                ]
            ],
            "index": [0]
        }
    }


    headers = {
        "Authorization": "Bearer " + ENDPOINT_KEY,
        "Content-Type": "application/json"
    }


    try:

        response = requests.post(
            ENDPOINT_URL,
            json=data,
            headers=headers,
            timeout=60
        )


        if response.status_code == 200:

            prediction = response.json()

            st.success("Prediction generated successfully!")

            st.metric(
                "Predicted Rings",
                round(float(prediction[0]), 2)
            )

        else:

            st.error(
                f"Azure ML returned status code {response.status_code}"
            )

            st.code(response.text)


    except Exception as e:

        st.error("Could not connect to Azure ML.")

        st.code(str(e))