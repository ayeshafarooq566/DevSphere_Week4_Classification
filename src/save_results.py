# DevSphere Internship Program
# Week 4 - Classification Model
# Save Prediction Results


# --------------------------------------------------
# 1. Import Libraries
# --------------------------------------------------

import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# 2. Load Dataset
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
# 3. Split Dataset
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

y_pred = model.predict(X_test)


# --------------------------------------------------
# 6. Calculate Accuracy
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)


# --------------------------------------------------
# 7. Create Results DataFrame
# --------------------------------------------------

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})


# --------------------------------------------------
# 8. Add Class Names
# --------------------------------------------------

results["Actual_Class"] = results["Actual"].map({
    0: data.target_names[0],
    1: data.target_names[1]
})

results["Predicted_Class"] = results["Predicted"].map({
    0: data.target_names[0],
    1: data.target_names[1]
})


# --------------------------------------------------
# 9. Save Results
# --------------------------------------------------

output_path = "outputs/predictions.csv"

results.to_csv(
    output_path,
    index=False
)


# --------------------------------------------------
# 10. Display Results
# --------------------------------------------------

print("=" * 60)
print("DEVSPHERE WEEK 4 - PREDICTION RESULTS")
print("=" * 60)

print("\nSaved Model:")
print(model_path)

print("\nAccuracy:")
print(f"{accuracy:.4f}")

print(f"\nAccuracy Percentage:")
print(f"{accuracy * 100:.2f}%")

print("\nPrediction Results:")
print(results.head(10))

print("\nTotal Predictions:")
print(len(results))

print("\nResults saved successfully:")
print(output_path)