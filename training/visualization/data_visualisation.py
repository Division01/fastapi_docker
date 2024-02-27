import pandas as pd

# Load the dataset
dataset_path = "dataset.csv"
data = pd.read_csv(dataset_path, header=0, delimiter=";")

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Display basic information about the dataset
print("\nDataset information:")
print(data.info())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())
print(
    "Position(s) of the missing value(s) : ",
    data[data["description"].isnull()].index.tolist(),
)

# Check the distribution of the target variable
print("\nDistribution of the target variable:")
print(data["fraudulent"].value_counts())

