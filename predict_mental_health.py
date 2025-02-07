import joblib
import pandas as pd

# Load the model and label encoders
model = joblib.load("mental_health_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Define new input data (example)
new_data = {
    "Age": 30,
    "Gender": "Male",
    "Country": "United States",
    "state": "TN",
    "self_employed": "No",
    "family_history": "No",
    "work_interfere": "Sometimes",
    "no_employees": "6-25",
    "remote_work": "No",
    "tech_company": "Yes",
    "benefits": "Yes",
    "care_options": "Not sure",
    "wellness_program": "No",
    "seek_help": "No",
    "anonymity": "Yes",
    "leave": "Somewhat easy",
    "mental_health_consequence": "No",
    "phys_health_consequence": "No",
    "coworkers": "Yes",
    "supervisor": "No",
    "mental_health_interview": "No",
    "phys_health_interview": "Yes",
    "mental_vs_physical": "Yes",
    "obs_consequence": "No"
}

# Convert new data to DataFrame
new_df = pd.DataFrame([new_data])

# Encode categorical variables using the saved label encoders
for column in new_df.select_dtypes(include=["object"]).columns:
    if column in label_encoders:
        new_df[column] = label_encoders[column].transform(new_df[column])

# Make a prediction
prediction = model.predict(new_df)
print("Predicted Treatment:", "Yes" if prediction[0] == 1 else "No")