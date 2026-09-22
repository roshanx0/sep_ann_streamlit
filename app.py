import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("employee_performance_ann.keras")

st.title("Employee Performance Predictor")

st.write("Enter the employee details below.")

# Inputs
training_hours = st.number_input(
    "Training Hours",
    min_value=0.0,
    max_value=100.0,
    value=5.0
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

# Prediction
if st.button("Predict Performance"):

    # Create input in the same order used during training
    input_data = np.array([[training_hours, attendance]])

    # Get prediction
    prediction = model.predict(input_data)

    # Convert probability to class
    if prediction[0][0] >= 0.5:
        result = "Good"
    else:
        result = "Needs Improvement"

    st.success(f"Predicted Performance: {result}")

    st.write(f"Prediction probability: {prediction[0][0]:.2f}")
