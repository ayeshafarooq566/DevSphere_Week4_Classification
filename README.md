# DevSphere Week 4 - Classification Model

## Machine Learning Classification Project

This project was developed as part of the **DevSphere Internship Program - Week 4 Machine Learning Task**.

The objective of this project is to implement a machine learning classification model using a classification dataset, train the model, generate predictions, and evaluate its performance using accuracy and other classification metrics.

---

## Project Objective

The main objectives of this project are:

- Use a classification dataset
- Prepare the dataset for machine learning
- Split the dataset into training and testing sets
- Train a classification model
- Generate predictions
- Calculate accuracy score
- Evaluate model performance
- Visualize the confusion matrix
- Save the trained machine learning model
- Save prediction results

---

## Dataset

### Breast Cancer Wisconsin Dataset

The project uses the **Breast Cancer Wisconsin Dataset** available through Scikit-learn.

### Dataset Information

- Number of samples: 569
- Number of features: 30
- Number of classes: 2
- Classification type: Binary Classification
- Training samples: 455
- Testing samples: 114

### Classes

The target variable contains two classes:

- Malignant
- Benign

---

## Machine Learning Model

### Logistic Regression

The classification algorithm used in this project is **Logistic Regression**.

Logistic Regression is a supervised machine learning algorithm commonly used for classification problems. It estimates the probability of an observation belonging to a particular class.

---

## Machine Learning Workflow

```text
Dataset
   ↓
Data Preparation
   ↓
Train-Test Split
   ↓
Logistic Regression
   ↓
Model Training
   ↓
Predictions
   ↓
Accuracy Evaluation
   ↓
Confusion Matrix
   ↓
Save Model
   ↓
Save Prediction Results