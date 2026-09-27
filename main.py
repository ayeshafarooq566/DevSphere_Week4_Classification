# DevSphere Internship Program
# Week 4 - Classification Model
# Logistic Regression Classification


# --------------------------------------------------
# 1. Import Libraries
# --------------------------------------------------

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib


# --------------------------------------------------
# 2. Load Classification Dataset
# --------------------------------------------------

data = load_breast_cancer()


# --------------------------------------------------
# 3. Create Features and Target
# --------------------------------------------------

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = pd.Series(
    data.target,
    name="target"
)


# --------------------------------------------------
# 4. Split Dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 5. Create Classification Model
# --------------------------------------------------

model = LogisticRegression(
    max_iter=10000,
    random_state=42
)


# --------------------------------------------------
# 6. Train the Model
# --------------------------------------------------

model.fit(X_train, y_train)


# --------------------------------------------------
# 7. Make Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 8. Calculate Accuracy
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)


# --------------------------------------------------
# 9. Display Model Information
# --------------------------------------------------

print("=" * 60)
print("DEVSPHERE WEEK 4 - CLASSIFICATION MODEL")
print("=" * 60)

print("\nDataset:")
print("Breast Cancer Wisconsin Dataset")

print("\nClassification Algorithm:")
print("Logistic Regression")

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

print("\nModel Training:")
print("Training completed successfully!")


# --------------------------------------------------
# 10. Display Predictions
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL PREDICTIONS")
print("=" * 60)

print("\nFirst 10 Actual Values:")
print(y_test.iloc[:10].values)

print("\nFirst 10 Predicted Values:")
print(y_pred[:10])


# --------------------------------------------------
# 11. Display Accuracy
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL ACCURACY")
print("=" * 60)

print(f"\nAccuracy Score: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")


# --------------------------------------------------
# 12. Display Classification Report
# --------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names
    )
)


# --------------------------------------------------
# 13. Save Trained Model
# --------------------------------------------------

model_path = "models/logistic_regression_model.pkl"

joblib.dump(
    model,
    model_path
)

print("\n" + "=" * 60)
print("MODEL SAVING")
print("=" * 60)

print(f"\nTrained model saved successfully:")
print(model_path)