import math
import os
import pandas as pd


def logMatrix(df):
    def safe_log(x):
        if x > 0:
            return math.log(x)
        else:
            raise ValueError("Encountered non-positive value")

    return df.applymap(safe_log)


def offsetlogMatrix(df):
    # # Find the smallest non-zero value in the dataframe
    # smallest = df[df > 0].min().min()
    #
    # # Calculate the offset as 0.5 * smallest
    # offset = 0.5 * smallest
    offset=1

    # Apply the log transformation with the calculated offset
    return df.applymap(lambda x: math.log(x + offset))


def standardizeMatrix(df):
    mean = df.stack().mean()  # Calculate mean of all elements
    stdev = df.stack().std()  # Calculate stdev of all elements

    return df.applymap(lambda x: (x - mean) / stdev)


def parse_csv_to_dataframe(file_path):
    return pd.read_csv(file_path, index_col=0)


folder_path = "old/mean_of_years_2025/country"
# Get list of CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Ensure the output directory exists
output_folder_path = "transformed_2025"
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Iterate through each CSV file
for file_name in csv_files:
    file_path = os.path.join(folder_path, file_name)
    df = parse_csv_to_dataframe(file_path)

    # Try logMatrix first, then fallback to logPseudoCountMatrix
    try:
        df = logMatrix(df)
        operation = "logtransform"
    except ValueError:
        df = offsetlogMatrix(df)
        operation = "offsetlog"

    df = standardizeMatrix(df)
    # Create the new file name with the operation included
    new_file_name = f"{os.path.splitext(file_name)[0]}_{operation}.csv"
    output_file_path = os.path.join(output_folder_path, new_file_name)
    df.to_csv(output_file_path)
