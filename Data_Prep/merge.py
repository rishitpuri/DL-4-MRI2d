import pandas as pd

# Load the CSV file containing the principal components into a DataFrame
genetic_data_df = pd.read_csv('pca_data.csv')

# Load the CSV file containing the image paths into another DataFrame
imaging_data_df = pd.read_csv('patient_data.csv')

# Merge the two DataFrames based on the 'Patient_ID' column
# This will only keep rows where the 'Patient_ID' exists in both DataFrames
merged_df = pd.merge(imaging_data_df, genetic_data_df, on='IID', how='inner')

# Display the number of matching patients
num_matching_patients = merged_df['IID'].nunique()

print(f"Number of unique patients with imaging data: {imaging_data_df['IID'].nunique()}")
print(f"Number of unique patients with genetic data: {genetic_data_df['IID'].nunique()}")

print(f"Number of patients with both imaging and genetic data: {num_matching_patients}")
print(f"Number of images available for training: {len(merged_df)}")

merged_df.to_csv('merged_data.csv', index=False)

