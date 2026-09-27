# DevSphere Internship Program
# Week 4 - Classification Model
# Model Evaluation and Confusion Matrix


# --------------------------------------------------
# 1. Import Libraries
# --------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import pandas as pd


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
# 4. Create and Train Model
# --------------------------------------------------

model = LogisticRegression(
    max_iter=10000,
    random_state=42
)

model.fit(X_train, y_train)


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
# 7. Create Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)


# --------------------------------------------------
# 8. Display Results
# --------------------------------------------------

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names
    )
)


# --------------------------------------------------
# 9. Create Confusion Matrix Visualization
# --------------------------------------------------

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=data.target_names,
    yticklabels=data.target_names
)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()


# --------------------------------------------------
# 10. Save Confusion Matrix
# --------------------------------------------------

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()