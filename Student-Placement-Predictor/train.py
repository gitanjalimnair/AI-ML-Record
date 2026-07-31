print("Train.py started")

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv("dataset/college_student_placement_dataset.csv")

# Display first few rows
print("First 5 rows of the dataset:")
print(data.head())

# Remove the College_ID column (not useful for prediction)
data = data.drop("College_ID", axis=1)

# Convert categorical columns to numerical values
data["Internship_Experience"] = data["Internship_Experience"].map({
    "Yes": 1,
    "No": 0
})

data["Placement"] = data["Placement"].map({
    "Yes": 1,
    "No": 0
})

# Separate features and target
X = data.drop("Placement", axis=1)
y = data["Placement"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
with open("placement_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully as placement_model.pkl")