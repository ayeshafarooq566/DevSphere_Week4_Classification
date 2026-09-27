# DevSphere Internship Program
# Week 4 - Classification Model
# Saved Model Prediction


# --------------------------------------------------
# 1. Import Libraries
# --------------------------------------------------

import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import pandas as pd


# --------------------------------------------------
# 2. Load Classification Dataset
# --------------------------------------------------

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = pd.Series(
    data.target,
    name="target"
)


# --------------------------------------------------
# 3. Create the Same Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 4. Load Saved Model
# --------------------------------------------------

model_path = "models/logistic_regression_model.pkl"

model = joblib.load(model_path)


# --------------------------------------------------
# 5. Make Predictions
# --------------------------------------------------

predictions = model.predict(X_test)


# --------------------------------------------------
# 6. Display Prediction Results
# --------------------------------------------------

print("=" * 60)
print("DEVSPHERE WEEK 4 - SAVED MODEL PREDICTION")
print("=" * 60)

print("\nSaved Model:")
print("Logistic Regression")

print("\nModel Loaded Successfully!")


print("\n" + "=" * 60)
print("PREDICTION RESULTS")
print("=" * 60)

print("\nFirst 10 Actual Values:")
print(y_test.iloc[:10].values)

print("\nFirst 10 Predicted Values:")
print(predictions[:10])


# --------------------------------------------------
# 7. Display Class Names
# --------------------------------------------------

print("\nClass Labels:")

for value, name in enumerate(data.target_names):
    print(f"{value} = {name}")


# --------------------------------------------------
# 8. Compare Predictions
# --------------------------------------------------

print("\n" + "=" * 60)
print("PREDICTION COMPARISON")
print("=" * 60)

comparison = pd.DataFrame({
    "Actual": y_test.iloc[:10].values,
    "Predicted": predictions[:10]
})

print(comparison)