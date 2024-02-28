import pickle
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.metrics import fbeta_score, confusion_matrix


# Function to clean the text data
def clean_text(text):
    if isinstance(text, str):
        # Remove special characters, numbers, and extra spaces
        text = re.sub(r"[^a-zA-Z\s\-/:]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    else:
        return ""


# Load the dataset
dataset_path = "./training/dataset.csv"
data = pd.read_csv(dataset_path, header=0, delimiter=";")

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
    X, y, test_size=0.2, random_state=42, stratify=y
)

parameters = {
    "C": 10,
    "gamma": "scale",
    "kernel": "rbf",
}
# Define the SVM model with specified parameters
svm_model = SVC(
    C=parameters["C"], gamma=parameters["gamma"], kernel=parameters["kernel"]
)

# Define the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

X_train = tfidf_vectorizer.fit_transform(X_train)
X_test = tfidf_vectorizer.transform(X_test)

oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)

svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

print("Classification Report:")
cr = classification_report(y_test, y_pred)
print(cr)

f2_score = fbeta_score(y_test, y_pred, beta=2)
print(f"F2-score: {f2_score}")

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# Define the pipeline with TF-IDF vectorization and SVM Classifier
pipeline = Pipeline(
    [
        ("tfidf", tfidf_vectorizer),
        ("svm", svm_model),
    ]
)

# Save the model, its score and its parameters
svm_pipeline = {
    "pipeline": pipeline,
    "parameters": parameters,
    "f2_score": f2_score,
    "confusion_matrix": cm,
    "classification_report": cr,
}
with open("./models/svm_pipeline.pkl", "wb") as f:
    pickle.dump(svm_pipeline, f)
