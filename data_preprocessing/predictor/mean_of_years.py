import os
import pandas as pd

# Path to the folder containing CSV files
parent_folder_path = 'predictors_raw'
output_folder_path = 'mean_of_years_2025'
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Function to parse CSV file into a DataFrame
def parse_csv_to_dataframe(file_path):
    df = pd.read_csv(file_path, index_col=0)
    return df

# Iterate over each subfolder
for folder_name in os.listdir(parent_folder_path):
    folder_path = os.path.join(parent_folder_path, folder_name)

    # Skip if not a folder
    if not os.path.isdir(folder_path):
        continue

    print(f"Processing folder: {folder_name}")

    # Get list of CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # Initialize sum DataFrame
    sum_df = None

    # Iterate through each CSV file
    for file_name in csv_files:
        file_path = os.path.join(folder_path, file_name)
        df = parse_csv_to_dataframe(file_path)
        if sum_df is None:
            sum_df = df
        else:
            sum_df += df

    # Calculate mean DataFrame
    mean_df = sum_df / len(csv_files)

    # Output or further processing with mean_df
    print(mean_df)
    name = os.path.join(output_folder_path, f'{folder_name}.csv')
    if not os.path.exists(name):
        mean_df.to_csv(name)