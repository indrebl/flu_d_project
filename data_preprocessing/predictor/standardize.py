import math
import os
import pandas as pd


def logMatrix(df):
    """Applies the natural logarithm to all positive elements in a DataFrame.

    Args:
        df (pandas.DataFrame): A DataFrame containing only positive numeric values.

    Returns:
        pandas.DataFrame: A DataFrame where each value is transformed by `math.log`.

    Raises:
        ValueError: If any element in the DataFrame is less than or equal to zero.
    """
    def safe_log(x):
        if x > 0:
            return math.log(x)
        else:
            raise ValueError("Encountered non-positive value")

    return df.applymap(safe_log)


def offsetlogMatrix(df):
    """Applies a natural log transformation to all elements in a DataFrame with an offset of 1.

    Args:
        df (pandas.DataFrame): A DataFrame containing numeric values.

    Returns:
        pandas.DataFrame: A DataFrame where each value is transformed by `math.log(x + 1)`.
    """
    offset = 1
    return df.applymap(lambda x: math.log(x + offset))


def standardizeMatrix(df):
    """Standardizes all elements in a DataFrame to have mean 0 and standard deviation 1.

    Args:
        df (pandas.DataFrame): A DataFrame containing numeric values.

    Returns:
        pandas.DataFrame: A DataFrame where each element is standardized.
    """
    mean = df.stack().mean()
    stdev = df.stack().std()

    return df.applymap(lambda x: (x - mean) / stdev)


folder_path = "mean_of_years_2025/country"
# Get list of CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Ensure the output directory exists
output_folder_path = "transformed_2025"
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Iterate through each CSV file
for file_name in csv_files:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path, index_col=0)

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
