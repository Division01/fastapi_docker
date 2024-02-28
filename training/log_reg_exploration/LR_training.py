import pandas as pd
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.pipeline import Pipeline

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


# Perform logistic regression

parameters = {"C": 10, "max_iter": 100, "penalty": "l1", "solver": "saga"}
lr_model = LogisticRegression(
    C=parameters["C"],
    max_iter=parameters["max_iter"],
    penalty=parameters["penalty"],
    solver=parameters["solver"],
)
lr_model.fit(X_train_resampled, y_train_resampled)


y_pred = lr_model.predict(X_test_tfidf)

print("Classification Report:")
cr = classification_report(y_test, y_pred)
print(cr)

f2_score = fbeta_score(y_test, y_pred, beta=2)
print(f"F2-score: {f2_score}")

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)


# Create a pipeline including TF-IDF vectorizer and logistic regression model
pipeline = Pipeline(
    [
        ("tfidf_vectorizer", tfidf_vectorizer),
        ("classifier", lr_model),
    ]
)

# Save the model, its score and its parameters
lr_pipeline = {
    "pipeline": pipeline,
    "parameters": parameters,
    "f2_score": f2_score,
    "confusion_matrix": cm,
    "classification_report": cr,
}
with open("./models/lr_pipeline.pkl", "wb") as f:
    pickle.dump(lr_pipeline, f)
