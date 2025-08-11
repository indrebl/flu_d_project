import pandas as pd
import os

if not os.path.exists('predictors_raw/cap_dist'):
    os.makedirs('predictors_raw/cap_dist')

# Read the CSV file
df = pd.read_csv("capdist.csv")

country_codes = {
    "China": "CHN",
    "Denmark": "DEN",
    "Sweden": "SWD",
    "France": "FRN",
    "Italy": "ITA",
    "Japan": "JPN",
    "Mexico": "MEX",
    "USA": "USA",
    "Netherlands":"NTH",
    "Turkey":"TUR",
    "Namibia":"NAM",
    "Canada":"CAN",
    "United_Kingdom": "UK",
    "Australia":"AUL"}

countries = ['China', 'Italy', 'Netherlands', 'Turkey', 'France', 'Denmark', "Sweden", 'Namibia', 'USA', 'Japan',
             'Canada', 'United_Kingdom', 'Mexico', 'Australia']

# Create an empty DataFrame with countries as both index and columns
matrix_df = pd.DataFrame(index=countries, columns=countries)

# Iterate through every value in the DataFrame
for index, row in matrix_df.iterrows():
    for column_name in matrix_df.columns:
        # Update the value in the DataFrame
        row_value = df[(df['ida'] == country_codes[index]) & (df['idb'] == country_codes[column_name])]
        new_value = row_value['kmdist'].iloc[0] if not row_value.empty else 0
        matrix_df.at[index, column_name] = new_value

print(matrix_df)
matrix_df.to_csv('predictors_raw/cap_dist/capital_distance_km.csv')