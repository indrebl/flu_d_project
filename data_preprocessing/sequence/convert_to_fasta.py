import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from dateutil import parser
import os


def autoRecognizeDate(date_str):
    """
    Attempt to recognize and format a date string to a Year-month-day format with variable precision.

    Args:
    date_str (str): The date string to be parsed and formatted.

    Returns:
    str or None: The formatted date string if recognized, otherwise None.
    """
    try:
        parsed_date = parser.parse(date_str)
        # Check if the day is explicitly stated or not present
        if date_str.count("-") == 1:
            return parsed_date.strftime('%Y-%m')
        elif date_str.count("-") == 0:
            return parsed_date.strftime('%Y')
        else:
            return parsed_date.strftime('%Y-%m-%d')
    except ValueError:
        return None  # Unable to recognize the date format


# Specify the output folder
output_folder = 'segment4_2025'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the Excel file into a pandas DataFrame
df = pd.read_excel("segment_4_2025.xlsx")

# Clean the DataFrame (remove extra spaces, convert lists from strings to actual lists, etc.)
df = df.applymap(lambda x: str(x).replace('[', '').replace(']', '').replace("'", ''))
# Process all collection_date values in the DataFrame
df['collection_date'] = df['collection_date'].apply(autoRecognizeDate)
df['country'] = df['country'].apply(lambda x: str(x).split(':')[0])
df[['strain', 'country']] = df[['strain', 'country']].applymap(lambda x: str(x).replace(' ', '_'))


# Group by the 'segment' column
grouped = df.groupby('segment')

# Create a FASTA file for each group
for segment, group in grouped:
    fasta_records = []
    for _, row in group.iterrows():
        record = SeqRecord(Seq(row['Seq']), id=f"{row['strain']}|{row['country']}|{row['collection_date']}", description="")
        fasta_records.append(record)
    SeqIO.write(fasta_records, os.path.join(output_folder, f"segment_{segment}.fasta"), "fasta")
