import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
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

# Vectorize text data using CountVectorizer followed by TfidfTransformer
count_vectorizer = CountVectorizer(max_features=5000)
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# Oversample the training data using SMOTE
oversampler = SMOTE()
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train_tfidf, y_train)

# Define a custom scorer for F2-score
f2_scorer = make_scorer(fbeta_score, beta=2)

# Define parameter grid for SGDClassifier
param_grid_sgd = {
    "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1],
    "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
}

# Perform GridSearchCV for SGDClassifier
sgd_model = SGDClassifier()
clf_sgd = GridSearchCV(
    sgd_model,
    param_grid=param_grid_sgd,
    cv=3,
    verbose=2,
    n_jobs=-1,
    scoring=f2_scorer,
)
clf_sgd.fit(X_train_resampled, y_train_resampled)

# Print the best parameters and evaluation metrics for SGDClassifier
print("\nBest parameters for SGDClassifier:")
print(clf_sgd.best_params_)
print(f"\nBest SGDClassifier CV score: {clf_sgd.best_score_}")
print(f"SGDClassifier Test score: {clf_sgd.score(X_test_tfidf, y_test)}")
