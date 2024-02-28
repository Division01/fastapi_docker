import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.model_selection import GridSearchCV

# Load the dataset
dataset_path = "dataset.csv"
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

# Filter that on relevant information only
data = data[data["word_count"] < 1200]


# Splitting the data into features (X) and target variable (y)
X = data["description"]
y = data["fraudulent"]

# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorize text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train = tfidf_vectorizer.fit_transform(X_train)
X_test = tfidf_vectorizer.transform(X_test)


# Define a function to calculate F2-score
def f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2)


# Define a custom scorer for GridSearchCV
f2_scorer = make_scorer(f2_score)

# Create a pipeline for Random Forest with class weighting
rf_pipeline = Pipeline(
    [
        ("rf", RandomForestClassifier()),
    ]
)

# Define parameter grid for GridSearchCV
param_grid = {
    "rf__n_estimators": [100, 200, 300],
    "rf__max_depth": [None, 10, 20],
    "rf__min_samples_split": [2, 5],
    "rf__min_samples_leaf": [1, 2],
    "rf__max_features": ["sqrt", "log2"],
}

from imblearn.over_sampling import SMOTE
from collections import Counter

oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)
# summarize the new class distribution
counter = Counter(y_train)
print(counter)

# Perform GridSearchCV
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, verbose=2, scoring=f2_scorer)
grid_search.fit(X_train, y_train)

# Get the best estimator and evaluate on test data
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print("Classification Report:")
print(classification_report(y_test, y_pred))

f2 = fbeta_score(y_test, y_pred, beta=2)
print(f"F2-score: {f2}")

"""
With SMOTE but no oversampler :
Best Parameters: {'rf__max_depth': None, 'rf__max_features': 'log2', 'rf__min_samples_leaf': 1, 'rf__min_samples_split': 2, 'rf__n_estimators': 300}
Classification Report:
              precision    recall  f1-score   support

           0       0.98      1.00      0.99      3403
           1       0.96      0.61      0.75       173

    accuracy                           0.98      3576
   macro avg       0.97      0.81      0.87      3576
weighted avg       0.98      0.98      0.98      3576

F2-score: 0.6608478802992519


With SMOTE : 
Best Parameters: {'rf__max_depth': None, 'rf__max_features': 'log2', 'rf__min_samples_leaf': 1, 'rf__min_samples_split': 2, 'rf__n_estimators': 200, 'sampling': RandomOverSampler()}
Classification Report:
              precision    recall  f1-score   support

           0       0.98      1.00      0.99      3403
           1       0.96      0.60      0.74       173

    accuracy                           0.98      3576
   macro avg       0.97      0.80      0.86      3576
weighted avg       0.98      0.98      0.98      3576

F2-score: 0.65



Without SMOTE : 
Best Parameters: {'rf__max_depth': 20, 'rf__max_features': 'log2', 'rf__min_samples_leaf': 2, 'rf__min_samples_split': 2, 'rf__n_estimators': 300, 'sampling': RandomOverSampler()}
Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.97      0.98      3403
           1       0.54      0.67      0.60       173

    accuracy                           0.96      3576
   macro avg       0.76      0.82      0.79      3576
weighted avg       0.96      0.96      0.96      3576

F2-score: 0.6387665198237885
"""
