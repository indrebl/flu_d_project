from Bio import SeqIO
import pandas as pd
import openpyxl

gb_file = "C:/Users/iblag/Downloads/segment4_2025.gb"

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

    # Handle missing strain and country information
    if "strain" not in gb_qual and "isolate" in gb_qual:
        gb_qual["strain"] = gb_qual.pop("isolate")
    if "strain" not in gb_qual:
        gb_qual["strain"] = ["unknown"]
    if "country" not in gb_qual and "geo_loc_name" in gb_qual:
        gb_qual["country"] = gb_qual.pop("geo_loc_name")
    if "country" not in gb_qual:
        gb_qual["country"] = ["unknown"]

    selected_keys = ["strain", "collection_date", "country", "length", "Seq", "description"]

    if "segment" in gb_qual:
        selected_keys.append("segment")
        gene_info = {key: gb_qual[key] for key in selected_keys}
        category = tuple(gene_info["strain"])

        if category in grouped_data:
            grouped_data[category].append(gene_info)
        else:
            grouped_data[category] = [gene_info]
    else:
        selected_keys = ["strain", "collection_date", "country", "length", "Seq", "description"]
        if all(key in gb_qual.keys() for key in selected_keys):
            gene_info = {key: gb_qual[key] for key in selected_keys}
            category = tuple(gene_info["strain"])

            if category in look_again:
                look_again[category].append(gene_info)
            else:
                look_again[category] = [gene_info]

# Segment 4 keywords and filtering
segments = {"4": ["hemagglutinin-esterase", "HEF"]}
filtered_data = {}

# Process grouped_data
for category, items in grouped_data.items():
    filtered_items = []
    for item in items:
        if "segment" in item and item["segment"][0] == "4":
            if item["length"] >= 0.8 * segment_length["4"]:
                Seq = item["Seq"]
                count = Seq.count("N") / len(Seq) * 100
                if count < 1:
                    filtered_items.append(item)
                else:
                    print(item["strain"], count)
        elif "segment" in item and item["segment"][0] != "4":
            description = item["description"]
            for key, values in segments.items():
                if any(value in description for value in values):
                    item["segment"] = key
                    if item["length"] >= 0.8 * segment_length["4"]:
                        Seq = item["Seq"]
                        count = Seq.count("N") / len(Seq) * 100
                        if count < 1:
                            filtered_items.append(item)
                        else:
                            print(item["strain"], count)
    filtered_data[category] = filtered_items

# Process look_again
for category, items in look_again.items():
    filtered_items = []
    for item in items:
        description = item["description"]
        for key, values in segments.items():
            if any(value in description for value in values):
                item["segment"] = key
                if item["length"] >= 0.8 * segment_length[key]:
                    Seq = item["Seq"]
                    count = Seq.count("N") / len(Seq) * 100
                    if count < 1:
                        filtered_items.append(item)
                    else:
                        print(item["strain"], count)
                break
    if category in filtered_data:
        filtered_data[category].extend(filtered_items)
    else:
        filtered_data[category] = filtered_items

# Convert filtered data to DataFrame
selected_keys = ["strain", "collection_date", "country", "segment", "Seq"]
dfs_filtered = [pd.DataFrame(items, columns=selected_keys) for items in filtered_data.values()]
df_filtered = pd.concat(dfs_filtered, keys=filtered_data.keys(), names=['Category'])

# Save DataFrame to Excel file
df_filtered.to_excel("segment_4_2025.xlsx")
