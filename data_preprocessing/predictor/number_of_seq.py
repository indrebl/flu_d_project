import os
import pandas as pd
from collections import Counter
from Bio import SeqIO


fasta_folder = "all_seq/final"
dest_folder = "seq_raw"


os.makedirs(dest_folder, exist_ok=True)

# Process each fasta
for fasta_file in os.listdir(fasta_folder):
    if not fasta_file.endswith(".fasta"):
        continue

    fasta_path = os.path.join(fasta_folder, fasta_file)

    # count countries
    country_counts = Counter()

    for record in SeqIO.parse(fasta_path, "fasta"):
        header = record.description
        parts = header.split("|")

        if len(parts) > 1:
            country = parts[1].strip()
            country_counts[country] += 1

    # Skip files with no country info
    if not country_counts:
        print(f"Skipped {fasta_file} (no country data)")
        continue

    # Create matrix
    present_countries = sorted(country_counts.keys())

    matrix_df = pd.DataFrame(
        0,
        index=present_countries,
        columns=present_countries,
        dtype=int
    )

    for country in present_countries:
        matrix_df.loc[country, :] = country_counts[country]

    transposed_matrix_df = matrix_df.T

    # save files
    name = fasta_file.replace(".cds.fasta", "")

    matrix_df.to_csv(
        os.path.join(dest_folder, f"{name}_num_of_seq_origin.csv")
    )

    transposed_matrix_df.to_csv(
        os.path.join(dest_folder, f"{name}_num_of_seq_destination.csv")
    )

    print(f"Processed {fasta_file}")
