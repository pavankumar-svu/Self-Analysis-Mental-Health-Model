import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("survey.csv")

# Handle missing values
df = df.fillna({
    "self_employed": "No",
    "work_interfere": "Don't know",
    "comments": "No comments"
})

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Save encoders for later use

# Define features (X) and target (y)
X = df.drop(columns=["treatment", "Timestamp", "comments"])  # Drop non-relevant columns
y = df["treatment"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
model = RandomForestClassifier(random_state=42, n_estimators=100)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# Save the model and label encoders
joblib.dump(model, "mental_health_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")


