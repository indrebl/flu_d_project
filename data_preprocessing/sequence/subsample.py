from collections import defaultdict
from Bio import SeqIO
import random

fasta_file = "all_seq/final/segment4.cds.fasta"
output_fasta = "all_seq/final/segment4.subsample.fasta"

# Group records by country
records_by_country = defaultdict(list)

for record in SeqIO.parse(fasta_file, "fasta"):
    header = record.description
    country = header.split("|")[1]
    records_by_country[country].append(record)

# Subsample and collect records
selected_records = []

for country, records in records_by_country.items():
    if len(records) > 20:
        selected = random.sample(records, 20)
    else:
        selected = records
    selected_records.extend(selected)

# Write new FASTA
SeqIO.write(selected_records, output_fasta, "fasta")

print(f"Wrote {len(selected_records)} sequences to {output_fasta}")
