# This script was inspired from Kaggle.
# It wasn't used as the necessity for nltk libraries and dependencies is considered to high for this project.
# It still is stored in this repo for insights.
# https://www.kaggle.com/code/seifwael123/real-fake-jobs-eda-modelling-99#Feature-Extraction

import pandas as pd
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_score,
)
import nltk
import spacy
import pickle
from scipy.sparse import hstack

# Load the dataset
dataset_path = "dataset.csv"
data = pd.read_csv(dataset_path, header=0, delimiter=";")

# Drop rows with empty descriptions
data = data.dropna(subset=["description"])


### DATA CLEANING
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = " ".join(tokens)
    return cleaned_text


data["description"] = data["description"].apply(clean_text)

### DATA NORMALIZATION
nlp = spacy.load("en_core_web_sm")


def normalize_text(text):
    # Tokenize the text and apply lemmatization
    doc = nlp(text)
    normalized_words = [token.lemma_ for token in doc]
    normalized_text = " ".join(normalized_words)
    return normalized_text


data["description"] = data["description"].apply(normalize_text)


### FEATURE EXTRACTION
text_columns = ["description"]

data["pos_features"] = data[text_columns].apply(
    lambda x: nltk.pos_tag(nltk.word_tokenize(x[0])), axis=1
)
data["pos_features"] = data["pos_features"].apply(
    lambda tags: " ".join(tag[1] for tag in tags)
)

X_train = data.drop("fraudulent", axis=1)
y_train = data["fraudulent"]


vectorizer = CountVectorizer(ngram_range=(1, 2))
text_matrix_train = vectorizer.fit_transform(X_train["description"])
pos_matrix_train = vectorizer.transform(X_train["pos_features"])
combined_matrix_train = hstack([text_matrix_train, pos_matrix_train])

with open("countvectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


# Splitting the data into features (X) and target variable (y)
X = combined_matrix_train
y = data["fraudulent"].values

# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

# Define the pipeline
pipeline = Pipeline([("sampling", RandomOverSampler()), ("svc", SVC())])

# Define the parameter grid
param_grid = {
    "svc__C": [0.1, 1, 10],
    "svc__kernel": ["linear", "rbf"],
    "svc__gamma": ["scale", "auto"],
}

# Perform grid search
# grid_search = GridSearchCV(
#     estimator=pipeline, param_grid=param_grid, cv=5, scoring="f1", verbose=2, n_jobs=-1
# )
# grid_search.fit(X_train, y_train)

# Get the best parameters and best score
# print("Best Parameters:", grid_search.best_params_)
# print("Best Score:", grid_search.best_score_)

# Assuming these are the best parameters found by GridSearchCV
best_C = 10
best_gamma = "scale"
best_kernel = "rbf"

# Initialize SVC with the best parameters
model = SVC(C=best_C, gamma=best_gamma, kernel=best_kernel)
model.fit(X_train, y_train)

cv_scores = cross_val_score(model, X_train, y_train, cv=5)
y_pred3 = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred3)
recall = recall_score(y_test, y_pred3)

precision = precision_score(y_test, y_pred3)
f1 = f1_score(y_test, y_pred3)
cm = confusion_matrix(y_test, y_pred3)
sns.heatmap(cm, annot=True, cmap="Blues", fmt="g")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("Cross-validation scores:", cv_scores)
print("Average cross-validation score:", cv_scores.mean())
print("Test set accuracy:", accuracy)
print("Test set precision:", precision)
print("Test set recall:", recall)
print("Test set F1 score:", f1)
