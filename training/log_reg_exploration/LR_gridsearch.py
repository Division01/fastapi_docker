import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, fbeta_score

# Load the dataset
dataset_path = "./training/dataset.csv"
data = pd.read_csv(dataset_path, header=0, delimiter=";")

# Add word_count column
data["word_count"] = data["description"].apply(lambda x: len(str(x).split()))


# Clean the description column
def clean_text(text):
    if isinstance(text, str):
        # Remove special characters, numbers, and extra spaces
        text = re.sub(r"[^a-zA-Z\s\-/:]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    else:
        return ""


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
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Oversample the training data using SMOTE
# oversampler = SMOTE()
# X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train_tfidf, y_train)
X_train_resampled, y_train_resampled = X_train_tfidf, y_train

# Define a custom scorer for F2-score
f2_scorer = make_scorer(fbeta_score, beta=2)

# Define parameter grid for logistic regression
param_grid_lr = {
    "penalty": ["l1", "l2", "elasticnet", "none"],
    "C": [0.001, 0.01, 0.1, 1, 10],
    "solver": ["lbfgs", "newton-cg", "liblinear", "sag", "saga"],
    "max_iter": [100, 500, 1000, 2500],
}

# Perform GridSearchCV for logistic regression
lr_model = LogisticRegression()
clf_lr = GridSearchCV(
    lr_model,
    param_grid=param_grid_lr,
    cv=3,
    verbose=2,
    n_jobs=-1,
    scoring=f2_scorer,
)
clf_lr.fit(X_train_resampled, y_train_resampled)

# Print the best parameters and evaluation metrics for logistic regression
print("\nBest parameters for Logistic Regression:")
print(clf_lr.best_params_)
print(f"\nBest Logistic Regression CV score: {clf_lr.best_score_}")
print(f"Logistic Regression Test score: {clf_lr.score(X_test_tfidf, y_test)}")


"""
With SMOTE :
Best parameters for Logistic Regression:
{'C': 10, 'max_iter': 100, 'penalty': 'l1', 'solver': 'saga'}

Best Logistic Regression CV score: 0.9923826006136358
Logistic Regression Test score: 0.6908267270668177


Without SMOTE : 
Best parameters for Logistic Regression:
{'C': 10, 'max_iter': 1000, 'penalty': 'l1', 'solver': 'saga'}

Best Logistic Regression CV score: 0.6121808108949033
Logistic Regression Test score: 0.6700379266750948
"""
