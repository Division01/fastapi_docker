import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

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
data['word_count'] = data['description'].apply(lambda x: len(str(x).split()))

# Add character count column
data['char_count'] = data['description'].apply(lambda x: len(str(x)))

# Clean the description column
data['clean_description'] = data['description'].apply(clean_text)

# Add word count after cleaning
data['word_count_clean'] = data['clean_description'].apply(lambda x: len(str(x).split()))

# Add character count after cleaning
data['char_count_clean'] = data['clean_description'].apply(lambda x: len(str(x)))

# Display the first few rows of the modified dataset
print("First few rows of the modified dataset:")
print(data.head())

# Display basic information about the modified dataset
print("\nModified dataset information:")
print(data.info())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Check the distribution of the target variable
print("\nDistribution of the target variable:")
print(data["fraudulent"].value_counts())


# Filter the dataset for descriptions with more than word_count words
for word_count in [700,800,900,1000,1100,1200]:
    long_descriptions = data[data['word_count'] > word_count]

    # Check the number of fraudulent descriptions in the filtered subset
    num_fraudulent_long_descriptions = long_descriptions[long_descriptions['fraudulent'] == 1].shape[0]
    num_non_fraudulent_long_descriptions = long_descriptions[long_descriptions['fraudulent'] == 0].shape[0]

    print(f"Number of fraudulent descriptions with more than {word_count} words:")
    print(f"{num_fraudulent_long_descriptions} fraud / {num_non_fraudulent_long_descriptions} non fraud.")


# Filter that on relevant information only
data = data[data['word_count'] < 1200]

# Plot the distribution of word counts for fraudulent and non-fraudulent samples
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='word_count', hue='fraudulent', bins=30, kde=True, alpha=0.7)
plt.title("Distribution of Word Counts by Fraudulent Label")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.legend(title="Fraudulent", loc='upper right')
plt.show()

# Separate fraudulent and non-fraudulent samples
fraudulent_data = data[data["fraudulent"] == 1]
non_fraudulent_data = data[data["fraudulent"] == 0]

# Plot histograms for word counts for fraudulent and non-fraudulent samples
plt.figure(figsize=(10, 6))

# Plot histogram for non-fraudulent samples
plt.hist(non_fraudulent_data['word_count'], bins=20, alpha=0.5, label='Non-Fraudulent', density=True)

# Plot histogram for fraudulent samples
plt.hist(fraudulent_data['word_count'], bins=20, alpha=0.5, label='Fraudulent', density=True)



plt.xlabel('Word Count')
plt.ylabel('Density')
plt.title('Distribution of Word Counts')
plt.legend()
plt.show()