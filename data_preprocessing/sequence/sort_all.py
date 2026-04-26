from Bio import SeqIO
import pandas as pd
import openpyxl
import os

gb_file = "complete.gb"

grouped_data = {}
look_again = {}
segment_length = {'1': 2364, '2': 2262, '3': 2133, '4': 1995, '5': 1659, '6': 1107, '7':820}

# Parse GenBank file and extract relevant information
for gb_record in SeqIO.parse(open(gb_file, "r"), "genbank"):

    gb_feature = gb_record.features[0]
    gb_qual = gb_feature.qualifiers

    length = len(gb_record.seq)
    description = gb_record.description
    gb_qual['length'] = length
    gb_qual['Seq'] = gb_record.seq
    gb_qual['description'] = description

    # Extract gene and product from CDS feature
    gene = []
    product = []
    for feat in gb_record.features:
        if feat.type == "CDS":
            gene = feat.qualifiers.get("gene", [])
            product = feat.qualifiers.get("product", [])
            break

    gb_qual["gene"] = gene
    gb_qual["product"] = product

    # Handle missing strain and country info
    if "strain" not in gb_qual and "isolate" in gb_qual:
        gb_qual["strain"] = gb_qual.pop("isolate")
    if "strain" not in gb_qual:
        gb_qual["strain"] = ["unknown"]
    if "country" not in gb_qual and "geo_loc_name" in gb_qual:
        gb_qual["country"] = gb_qual.pop("geo_loc_name")
    if "country" not in gb_qual:
        gb_qual["country"] = ["unknown"]

    selected_keys = ["strain", "collection_date", "country", "length", "Seq", "description", "gene", "product"]

    # Entries with segment
    if "segment" in gb_qual:
        selected_keys.append("segment")
        gene_info = {key: gb_qual.get(key, []) for key in selected_keys}
        category = tuple(gene_info["strain"])

        grouped_data.setdefault(category, []).append(gene_info)

    # Entries without segment
    else:
        selected_keys = ["strain", "collection_date", "country", "length", "Seq", "description", "gene", "product"]
        if all(key in gb_qual.keys() for key in selected_keys):
            gene_info = {key: gb_qual.get(key, []) for key in selected_keys}
            category = tuple(gene_info["strain"])

            look_again.setdefault(category, []).append(gene_info)


# Segment keywords
segments = {"1":["PB2"], "2":["PB1"],"3": ["P3"], "4": ["hemagglutinin-esterase", "HEF"], "5":["nucleocapsid",
            "nucleoprotein","NP"], "6":["P42"], "7":["nonstructural protein","NS1"]}


filtered_data = {}

for category, items in grouped_data.items():
    filtered_items = []

    for item in items:
        text_block = item["description"]
        if item["gene"]:
            text_block += " " + " ".join(item["gene"])
        if item["product"]:
            text_block += " " + " ".join(item["product"])

        kept = False

        # Trust declared segment first (for all segments)
        country = item.get("country", [""])[0].lower()
        if "sweden" in country and "hef" in text_block.lower():
            item["segment"] = ["4"]
            if item["length"] >= 0.8 * segment_length["4"]:
                count = item["Seq"].count("N") / len(item["Seq"]) * 100
                if count < 1:
                    filtered_items.append(item)
                    kept = True

        if not kept and "segment" in item:
            seg = item["segment"][0]
            if seg in segment_length:
                if item["length"] >= 0.8 * segment_length[seg]:
                    count = item["Seq"].count("N") / len(item["Seq"]) * 100
                    if count < 1:
                        filtered_items.append(item)
                        kept = True

        # Fallback to keyword reassignment
        if not kept:
            for key, values in segments.items():
                if any(value.lower() in text_block.lower() for value in values):
                    item["segment"] = [key]
                    if item["length"] >= 0.8 * segment_length[key]:
                        count = item["Seq"].count("N") / len(item["Seq"]) * 100
                        if count < 1:
                            filtered_items.append(item)
                    break

    filtered_data[category] = filtered_items



# Process look_again
for category, items in look_again.items():
    filtered_items = []

    for item in items:
        text_block = item["description"]
        if item["gene"]:
            text_block += " " + " ".join(item["gene"])
        if item["product"]:
            text_block += " " + " ".join(item["product"])

        for key, values in segments.items():
            if any(value.lower() in text_block.lower() for value in values):
                item["segment"] = [key]
                if item["length"] >= 0.8 * segment_length[key]:
                    count = item["Seq"].count("N") / len(item["Seq"]) * 100
                    if count < 1:
                        filtered_items.append(item)
                break

    filtered_data.setdefault(category, []).extend(filtered_items)



# Convert to DataFrame

selected_keys = ["strain", "collection_date", "country", "segment", "Seq"]

dfs = [
    pd.DataFrame(items, columns=selected_keys)
    for items in filtered_data.values()
    if items
]

df_filtered = pd.concat(dfs, ignore_index=True)

# Clean country column
df_filtered["country"] = df_filtered["country"].apply(
    lambda x: x[0].split(":")[0].strip() if isinstance(x, list) else str(x).split(":")[0].strip()
)

# Normalize segment column (list to string)
df_filtered["segment"] = df_filtered["segment"].apply(
    lambda x: x[0] if isinstance(x, list) else x
)

# Write one Excel file per segment
for seg in sorted(df_filtered["segment"].unique()):
    seg_df = df_filtered[df_filtered["segment"] == seg]
    outfile = f"segment_{seg}_fix.xlsx"
    seg_df.to_excel(outfile, index=False)
    print(f"Wrote {outfile} ({len(seg_df)} records)")


#COMPLETE STRAINS

complete_dir = "complete_strains_only_fix"
os.makedirs(complete_dir, exist_ok=True)

df_complete = df_filtered.copy()

# Normalize strain
def normalize_strain(x):
    if isinstance(x, list):
        return x[0]
    return str(x)

df_complete["strain"] = df_complete["strain"].apply(normalize_strain)

# Normalize segment
def normalize_segment(x):
    if isinstance(x, list):
        return str(x[0])
    x = str(x).strip()
    if x.startswith("[") and x.endswith("]"):
        return x.strip("[]").strip()
    return x

df_complete["segment"] = df_complete["segment"].apply(normalize_segment)

# Required segments
required_segments = {"1", "2", "3", "4", "5", "6", "7"}

# Find strains with ALL 7 DISTINCT segments
strain_segments = (
    df_complete
    .groupby("strain")["segment"]
    .apply(set)
)

complete_strains = strain_segments[
    strain_segments == required_segments
].index

print(f"Found {len(complete_strains)} strains with all 7 segments")

#  Keep only complete strains
df_complete = df_complete[df_complete["strain"].isin(complete_strains)]

# Write per-segment Excel files
for seg in sorted(required_segments):
    seg_df = df_complete[df_complete["segment"] == seg]

    if not seg_df.empty:
        outfile = os.path.join(
            complete_dir,
            f"segment_{seg}_complete_strains.xlsx"
        )
        seg_df.to_excel(outfile, index=False)
        print(f"Wrote {outfile} ({len(seg_df)} records)")
