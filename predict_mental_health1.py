import joblib
import pandas as pd

# Load the trained model and label encoders
model = joblib.load("mental_health_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Handle missing class_labels.pkl
try:
    class_labels = joblib.load("class_labels.pkl")
except FileNotFoundError:
    class_labels = {}
    print("Warning: 'class_labels.pkl' not found. Using an empty dictionary.")

# Define input features (keeping 'Age', 'self_employed', 'Marriage' and removing unnecessary ones)
input_features = [
    "Age", "Gender", "Country", "self_employed", "Marriage", 
    "no_employees", "remote_work", "tech_company"
]

# Collect user input
new_data = {}
for feature in input_features:
    value = input(f"Enter {feature}: ").strip()
    
    # Convert Age to integer
    if feature == "Age":
        try:
            value = int(value)
        except ValueError:
            print("Invalid input for Age. Setting to default value 30.")
            value = 30  # Default age if invalid input
    
    new_data[feature] = value

# Convert input to DataFrame
new_df = pd.DataFrame([new_data])

# Encode categorical variables while handling unseen labels
for column in new_df.select_dtypes(include=["object"]).columns:
    if column in label_encoders:
        known_labels = set(label_encoders[column].classes_)
        
        # Replace unseen labels with the most common category from training
        new_df[column] = new_df[column].apply(
            lambda x: x if x in known_labels else label_encoders[column].classes_[0]
        )
        
        new_df[column] = label_encoders[column].transform(new_df[column])

# Ensure DataFrame matches model's expected features
expected_features = model.feature_names_in_  # Fetch trained model's feature names
for feature in expected_features:
    if feature not in new_df.columns:
        new_df[feature] = 0  # Add missing feature with default value

# Reorder columns to match training data
new_df = new_df[expected_features]

# Make a prediction
prediction = model.predict(new_df)[0]

# Convert numerical prediction back to class label
predicted_label = class_labels.get(prediction, "Unknown")

print("\nPredicted Mental Health Condition:", predicted_label)
