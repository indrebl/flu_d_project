import pandas as pd
import os
dest_folder = 'predictors_raw/gdp_per_capita_destination'
ori_folder = 'predictors_raw/gdp_per_capita_origin'


if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)
if not os.path.exists(ori_folder):
    os.makedirs(ori_folder)

# Read the CSV file
df = pd.read_csv("gdp.csv")
# Replace "United States of America" with "USA"
df['Area'] = df['Area'].replace('United States of America', 'USA')
df['Area'] = df['Area'].replace('China, mainland', 'China')
df['Area'] = df['Area'].replace('Türkiye', 'Turkey')
df['Area'] = df['Area'].replace("Netherlands (Kingdom of the)", 'Netherlands')
df['Area'] = df['Area'].replace("United Kingdom of Great Britain and Northern Ireland", 'United_Kingdom')

# Filter the DataFrame to get only GDP per capita data
df_gdp_per_capita = df[(df['Element'] == 'Value US$ per capita') & (df['Item'] == 'Gross Domestic Product')]
# df_gdp_per_capita = df

# Get unique countries
countries = ['China', 'Italy', 'Netherlands', 'Turkey', 'France', 'Denmark', "Sweden", 'Namibia', 'USA', 'Japan',
             'Canada', 'United_Kingdom', 'Mexico', 'Australia']

# Create an empty DataFrame with countries as both index and columns
matrix_df = pd.DataFrame(index=countries, columns=countries)

# Iterate over each year and fill in GDP per capita values in the matrix
for year in df_gdp_per_capita['Year'].unique():
    gdp_year = df_gdp_per_capita[df_gdp_per_capita['Year'] == year]
    for country in countries:
        gdp_country = gdp_year[gdp_year['Area'] == country]['Value'].values
        if len(gdp_country) == 1:  # Ensure the country has GDP per capita data for this year
            matrix_df.loc[country, :] = gdp_country[0]
    transposed_matrix_df = matrix_df.T
    print(matrix_df)
    matrix_df.to_csv(os.path.join(ori_folder, f'gdp_per_capita_{year}.csv'))
    transposed_matrix_df.to_csv(os.path.join(dest_folder, f'gdp_per_capita_dest_{year}.csv'))
