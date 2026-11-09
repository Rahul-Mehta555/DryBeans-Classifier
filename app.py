import joblib
import pandas as pd
import streamlit as st

@st.cache_resource
def load_artifact():
    return joblib.load("agriculture_model.joblib")

artifact = load_artifact()
model = artifact["model"]
label_encoder = artifact["label_encoder"]
feature_names = artifact["feature_names"]
defaults = artifact["defaults"]

st.title("Dry Bean Classifier")
st.write("Enter the bean measurements to predict its class.")

with st.form("prediction_form"):
    values = {}
    columns = st.columns(2)

    for index, feature in enumerate(feature_names):
        with columns[index % 2]:
            values[feature] = st.number_input(
                feature,
                value=float(defaults[feature]),
                format="%.6f",
            )

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame([values], columns=feature_names)
    prediction = model.predict(input_data)
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    st.success(f"Predicted class: **{predicted_class}**")