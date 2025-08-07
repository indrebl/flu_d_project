import os
import re
from Bio import SeqIO
from collections import Counter
import baltic as bt

# Define the file path to the segment 4 alignment file
fasta_file = 'C:/Users/iblag/Documents/github/flu_D/influ_D_project_v2/segment4_2025/segment4_alignment.fasta'
# Define stop codons
stops = ['TGA', 'TAG', 'TAA']

# Extract the segment name
seg = 'segment4'

# Read the sequences from the file
seqs = {}  # Dictionary to hold segment sequences

for seq in SeqIO.parse(fasta_file, 'fasta'):  # Parse the FASTA file
    sequence = str(seq.seq)
    seqs[seq.id] = sequence  # Add sequence to the dictionary
    L = len(sequence)  # Get the sequence length

# Generate the consensus sequence
consensus = ''.join([Counter([seqs[strain][i] for strain in seqs]).most_common(1)[0][0] for i in range(L)])


# Remove columns where the consensus has gaps
strip_gaps = [m.start() for m in re.finditer('-', consensus)]  # Identify gap positions
consensus = ''.join([consensus[i] for i in range(len(consensus)) if i not in strip_gaps])  # Strip gaps from consensus

for seq in seqs:  # Remove gaps from all sequences
    seqs[seq] = ''.join([seqs[seq][i] for i in range(len(seqs[seq])) if i not in strip_gaps])

# Find the coding region (CDS)
start_codon = consensus.find('ATG')  # Find the start codon
stop_codon = None  # Initialize stop codon position

for i in range(start_codon, len(consensus), 3):  # Iterate over triplets in the consensus
    codon = consensus[i:i + 3]  # Extract the codon
    if codon in stops:  # Check for stop codon
        stop_codon = i + 3  # Set the stop codon position
        break

# Write the coding sequences (CDS) to a new file
out_cds_file = fasta_file.replace('.fasta', '.cds.fasta')
with open(out_cds_file, 'w') as out:
    for seq in seqs:
        out.write(f'>{seq}\n{seqs[seq][start_codon:stop_codon]}\n')  # Write the CDS

# Extract collection dates and write them to a file
out_dates_file = fasta_file.replace('.fasta', '.dates.txt')
with open(out_dates_file, 'w') as out_dates:
    for seq in seqs:
        date_string = seq.split('|')[-1]  # Extract date from sequence ID
        decD = bt.decimalDate(date_string, variable=True)  # Get decimal date & precision
        out_dates.write(f'{seq}\t{decD}\n')  # Write sequence ID, decimal date, and precision

