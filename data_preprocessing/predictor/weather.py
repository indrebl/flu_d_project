import os
import pandas as pd
import numpy as np

folder_path = "weather_1991-2023"
# Get list of CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Ensure the output directory exists
output_folder_path = "weather_2025"
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

offset=5

precipitation= {}
temperature= {}
# Iterate through each CSV file
for country_file in csv_files:
    file_path = os.path.join(folder_path, country_file)
    df = pd.read_csv(file_path, index_col=0)
    country = country_file.split('.')[0]
    precipitation[country] = sum(df['Precipitation'])
    temperature[country] = np.mean(df['Average Mean Surface Air Temperature']) + offset

countries= precipitation.keys()
#create empty matrix
print(countries)
matrix_temp = pd.DataFrame(index=countries, columns=countries)
matrix_prec = pd.DataFrame(index=countries, columns=countries)

for country in countries:
    matrix_temp.loc[country, :] = temperature[country]
    matrix_prec.loc[country, :] = precipitation[country]

print(matrix_temp)
print(matrix_prec)

transposed_matrix_temp = matrix_temp.T
matrix_temp.to_csv(os.path.join(output_folder_path,'mean_temperatures_origin.csv'))
transposed_matrix_temp.to_csv(os.path.join(output_folder_path,'mean_temperatures_destination.csv'))

transposed_matrix_prec = matrix_prec.T
matrix_prec.to_csv(os.path.join(output_folder_path,'yearly_precipitation_origin.csv'))
transposed_matrix_prec.to_csv(os.path.join(output_folder_path,'yearly_precipitation_destination.csv'))