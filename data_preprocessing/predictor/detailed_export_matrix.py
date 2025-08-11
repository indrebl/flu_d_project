import pandas as pd
import os

if not os.path.exists('predictors_raw/swine_trade_exp'):
    os.makedirs('predictors_raw/swine_trade_exp')

# Read the CSV file
df = pd.read_csv("swine_export_FAO.csv")

# Replace "United States of America" with "USA"
df['Reporter Countries'] = df['Reporter Countries'].replace('United States of America', 'USA')
df['Reporter Countries'] = df['Reporter Countries'].replace('China, mainland', 'China')
df['Reporter Countries'] = df['Reporter Countries'].replace('Türkiye', 'Turkey')
df['Reporter Countries'] = df['Reporter Countries'].replace("Netherlands (Kingdom of the)", 'Netherlands')
df['Reporter Countries'] = df['Reporter Countries'].replace("United Kingdom of Great Britain and Northern Ireland",
                                                            'United_Kingdom')

df['Partner Countries'] = df['Partner Countries'].replace('United States of America', 'USA')
df['Partner Countries'] = df['Partner Countries'].replace('China, mainland', 'China')
df['Partner Countries'] = df['Partner Countries'].replace('Türkiye', 'Turkey')
df['Partner Countries'] = df['Partner Countries'].replace("Netherlands (Kingdom of the)", 'Netherlands')
df['Partner Countries'] = df['Partner Countries'].replace("United Kingdom of Great Britain and Northern Ireland",
                                                          'United_Kingdom')


# Get a list of all unique countries mentioned in both reporter and partner sides
all_countries = all_countries = ['China', 'Italy', 'Netherlands', 'Turkey', 'France', 'Denmark', "Sweden",
                                 'Namibia', 'USA', 'Japan', 'Canada', 'United_Kingdom', 'Mexico', 'Australia']


# Filter the DataFrame to get only 'Export Quantity' of 'Swine / pigs'
df_export_cattle = df[(df['Element'] == 'Export Quantity') & (df['Item'] == 'Swine / pigs') & (df['Unit'] == 'An')]

for year in df_export_swine['Year'].unique():
    export_year = df_export_swine[df_export_swine['Year'] == year]

    # Reset the index to avoid duplicate entries
    export_year = export_year.reset_index(drop=True)

    # Create an empty DataFrame with all countries as rows and columns
    export_matrix = pd.DataFrame(index=all_countries, columns=all_countries)

    # Fill the matrix with export quantities for the current year
    for index, row in export_year.iterrows():
        reporter_country = row['Reporter Countries']
        partner_country = row['Partner Countries']
        export_quantity = row['Value']
        export_matrix.loc[reporter_country, partner_country] = export_quantity
        if partner_country == reporter_country:
            export_matrix.loc[reporter_country, partner_country] = 0

    # Fill missing values with 0 and infer objects
    export_matrix = export_matrix.fillna(0).infer_objects()

    # Remove the names of the index and columns
    export_matrix.index.name = None
    export_matrix.columns.name = None

    print(export_matrix)
    export_matrix.to_csv(os.path.join('predictors_raw/swine_trade_exp', f'swine_trade_export_{year}.csv'))