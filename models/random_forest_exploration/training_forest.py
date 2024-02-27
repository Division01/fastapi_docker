import pickle
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from imblearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score


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
dataset_path = "dataset.csv"
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

# Define the Random Forest model with specified parameters
rf_model = RandomForestClassifier(
    max_depth=20,
    max_features="log2",
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=200,
)

# Define the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Define the pipeline with data preprocessing, TF-IDF vectorization, RandomOverSampler, and Random Forest Classifier
pipeline = Pipeline(
    [
        ("clean_text", FunctionTransformer(clean_text)),
        ("tfidf", tfidf_vectorizer),
        ("sampling", RandomOverSampler()),
        ("rf", rf_model),
    ]
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
f2 = fbeta_score(y_test, y_pred, beta=2)
print(f"F2-score: {f2}")

# Once trained, save the model to a pickle file
with open("trained_rf_pipeline.pkl", "wb") as file:
    pickle.dump(pipeline, file)
