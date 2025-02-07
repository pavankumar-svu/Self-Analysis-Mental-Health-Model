import joblib
import pandas as pd
import streamlit as st

# Load the model and label encoders
model = joblib.load("mental_health_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Handle missing class_labels.pkl
try:
    class_labels = joblib.load("class_labels.pkl")
except FileNotFoundError:
    class_labels = {}
    st.warning("Warning: 'class_labels.pkl' not found. Using an empty dictionary.")

# Define the expected features (only the required ones)
expected_features = [
    "Age", "Gender", "Country", "self_employed", "Marriage", "no_employees", 
    "remote_work", "tech_company"
]  # Ensuring only required features

# Streamlit UI for user input
st.title("Mental Health Prediction")

new_data = {}
for feature in expected_features:
    new_data[feature] = st.text_input(f"Enter {feature}:", "")

# Convert new data to DataFrame
new_df = pd.DataFrame([new_data])

# Encode categorical variables using the saved label encoders
for column in new_df.select_dtypes(include=["object"]).columns:
    if column in label_encoders:
        known_labels = set(label_encoders[column].classes_)
        new_df[column] = new_df[column].apply(lambda x: x if x in known_labels else "Unknown")
        new_df[column] = label_encoders[column].transform(new_df[column])

# Ensure the DataFrame only has expected features
new_df = new_df[expected_features]

# Make a prediction
if st.button("Predict"):
    prediction = model.predict(new_df)[0]
    predicted_label = class_labels.get(prediction, "Unknown")
    st.success(f"Predicted Mental Health Condition: {predicted_label}")
