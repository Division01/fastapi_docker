import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load the dataset
dataset_path = "./training/dataset.csv"
data = pd.read_csv(dataset_path, header=0, delimiter=";")


# Function to clean the text data
def clean_text(text):
    if isinstance(text, str):
        # Remove special characters, numbers, and extra spaces
        text = re.sub(r"[^a-zA-Z\s\-/:]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    else:
        return ""


# Add word_count column
data["word_count"] = data["description"].apply(lambda x: len(str(x).split()))

# Clean the description column
data["description"] = data["description"].apply(clean_text)

# Filter on relevant information only
data = data[data["word_count"] < 1200]


# Splitting the data into features (X) and target variable (y)
X = data["description"]
y = data["fraudulent"]

# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorize text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000
)  # You can adjust max_features as needed
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

from imblearn.over_sampling import SMOTE
from collections import Counter

# Instantiate the SMOTE oversampler
oversampler = SMOTE()

# Oversample the training data
X_train_tfidf, y_train = oversampler.fit_resample(X_train_tfidf, y_train)

# Summarize the new class distribution
counter = Counter(y_train)
print(counter)

# Define the parameter grids for each model
param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
}

param_grid_lr = {
    "penalty": ["l1", "l2", "elasticnet", "none"],
    "C": [0.001, 0.01, 0.1, 1, 10],
    "solver": ["lbfgs", "newton-cg", "liblinear", "sag", "saga"],
    "max_iter": [100, 500, 1000, 2500],
}

param_grid_svm = {
    "C": [0.001, 0.01, 0.1, 1, 10],
    "kernel": ["linear", "rbf", "poly", "sigmoid"],
    "gamma": ["scale", "auto"],
}

# Create a dictionary of models and their corresponding parameter grids
models = {
    "Random Forest": (RandomForestClassifier(), param_grid_rf),
    "Logistic Regression": (LogisticRegression(), param_grid_lr),
    "Support Vector Machine": (SVC(), param_grid_svm),
}

best_results = {}

from sklearn.metrics import make_scorer, fbeta_score

# Define a custom scorer for F2-score
f2_scorer = make_scorer(fbeta_score, beta=2)


# Perform GridSearchCV for each model
for model_name, (model, param_grid) in models.items():
    clf = GridSearchCV(
        model, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1, scoring=f2_scorer
    )
    clf.fit(X_train_tfidf, y_train)

    # Print the best parameters and evaluation metrics for each model
    print(f"\nBest parameters for {model_name}:")
    print(clf.best_params_)
    print(f"\nBest {model_name} CV score: {clf.best_score_}")
    score = clf.score(X_test_tfidf, y_test)
    print(f"\n{model_name} Test score: {score}")
    best_results[model_name] = {
        "Parameters": clf.best_params_,
        "Best CV score": clf.best_score_,
        "Test score": score,
    }

print(best_results)

"""
{'Random Forest': {'Parameters': {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 100}, 'Best CV score': 0.9938711569126154, 'Test score': 0.6150793650793651},
'Logistic Regression': {'Parameters': {'C': 10, 'max_iter': 100, 'penalty': 'l1', 'solver': 'saga'}, 'Best CV score': 0.9922941389529524, 'Test score': 0.691609977324263}, 
'Support Vector Machine': {'Parameters': {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}, 'Best CV score': 0.9991472199005283, 'Test score': 0.6592689295039165}}
"""
