import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
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
oversampler = SMOTE()
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train_tfidf, y_train)

# Define a custom scorer for F2-score
f2_scorer = make_scorer(fbeta_score, beta=2)

# Define parameter grid for SVM
param_grid_svm = {
    "C": [0.001, 0.01, 0.1, 1, 10],
    "kernel": ["linear", "rbf", "poly", "sigmoid"],
    "gamma": ["scale", "auto"],
}

# Perform GridSearchCV for SVM
svm_model = SVC()
clf_svm = GridSearchCV(
    svm_model,
    param_grid=param_grid_svm,
    cv=3,
    verbose=2,
    n_jobs=-1,
    scoring=f2_scorer,
)
clf_svm.fit(X_train_resampled, y_train_resampled)

# Print the best parameters and evaluation metrics for SVM
print("\nBest parameters for SVM:")
print(clf_svm.best_params_)
print(f"\nBest SVM CV score: {clf_svm.best_score_}")
print(f"SVM Test score: {clf_svm.score(X_test_tfidf, y_test)}")


"""
Best parameters for SVM:
{'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}

Best SVM CV score: 0.9985584812273934
SVM Test score: 0.6592689295039165
"""
