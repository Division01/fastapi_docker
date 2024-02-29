import pickle
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


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

# Optional: Filter on relevant information only
# data = data[data["word_count"] < 1200]

# Splitting the data into features (X) and target variable (y)
X = data["description"]
y = data["fraudulent"]

# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

parameters = {
    "alpha": 0.01,
    "loss": "perceptron",
}

# Define the SGDClassifier model with specified parameters
sgd_model = SGDClassifier(alpha=parameters["alpha"], loss=parameters["loss"])

# Optional: Use CountVectorizer
count_vectorizer = CountVectorizer(max_features=5000)
X_train = count_vectorizer.fit_transform(X_train)
X_test = count_vectorizer.transform(X_test)

# Define the TF-IDF vectorizer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train)
X_test_tfidf = tfidf_transformer.transform(X_test)


# Oversample the training data using SMOTE
oversampler = SMOTE()
X_train_tfidf, y_train = oversampler.fit_resample(X_train_tfidf, y_train)

# Train the SGDClassifier model
sgd_model.fit(X_train_tfidf, y_train)
y_pred = sgd_model.predict(X_test_tfidf)

# Print classification report
print("Classification Report:")
cr = classification_report(y_test, y_pred)
print(cr)

# Calculate F2-score
f2_score = fbeta_score(y_test, y_pred, beta=2)
print(f"F2-score: {f2_score}")

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# Define the pipeline with TF-IDF vectorization and SGDClassifier
pipeline = Pipeline(
    [
        ("vectorize", count_vectorizer),
        ("tfidf", tfidf_transformer),
        ("sgd", sgd_model),
    ]
)

# Save the model, its score, and its parameters
sgd_pipeline = {
    "pipeline": pipeline,
    "parameters": parameters,
    "f2_score": f2_score,
    "confusion_matrix": cm,
    "classification_report": cr,
}
with open("./models/sgd_pipeline.pkl", "wb") as f:
    pickle.dump(sgd_pipeline, f)
