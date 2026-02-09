"""
Created 02.10.2024 by Christina Drotenko

Purpose of code: data processing
Input: raw data
Output: processed data

"""

#%% Importing data

# Imports
import pandas as pd
import numpy as np

#%%

# Read FABIO data
# INSERT FILE PATH BELOW:
#file_path = r"FILEPATH"

E_gwp = pd.read_csv(file_path + r"\E_gwp_2020.csv", delimiter=',')
fd_codes = pd.read_csv(file_path + r"\fd_codes.csv", delimiter=',')
fd_labels = pd.read_csv(file_path + r"\fd_labels.csv", delimiter=',')
gwp_names = pd.read_csv(file_path + r"\gwp_names.csv", delimiter=',')
io_codes = pd.read_csv(file_path + r"\io_codes.csv", delimiter=',')
io_labels = pd.read_csv(file_path + r"\io_labels.csv", delimiter=',')
items = pd.read_csv(file_path + r"\items.csv", delimiter=',')
regions = pd.read_csv(file_path + r"\regions.csv", delimiter=',')
x = pd.read_csv(file_path + r"\x_2020v3.csv", delimiter=',')
Y = pd.read_csv(file_path + r"\Y_2020v3.csv", delimiter=',')
Z = pd.read_csv(file_path + r"\Z_2020v3.csv", delimiter=',')

# Read Baoxiao et al.'s land use extension
land_use = pd.read_csv(file_path + r"\2020_cropgrids_hilda2019_cultivated_E.csv", delimiter=',')

# Read continent/country concordance
continents = pd.read_excel(file_path + r"\Concordance.xlsx", sheet_name="Concordance - GLEAM")

#%% Allocating concordance continents label to FABIO countries label

# Create dataframes that extract from the "continents" dataframe from the concordance matrix and allocate for each continent
east_southeast_asia = continents[continents['continent'] == 'East Asia and Southeast Asia']
east_europe = continents[continents['continent'] == 'Eastern Europe']
latam_caribbean = continents[continents['continent'] == 'Latin America and the Caribbean']
east_north_africa = continents[continents['continent'] == 'Near East and North Africa']
north_america = continents[continents['continent'] == 'North America']
oceania = continents[continents['continent'] == 'Oceania']
rest_of_world = continents[continents['continent'] == 'RoW']
russian_fed = continents[continents['continent'] == 'Russian Federation']
south_asia = continents[continents['continent'] == 'South Asia']
sub_saharan_africa = continents[continents['continent'] == 'Sub-Saharan Africa']
west_europe = continents[continents['continent'] == 'Western Europe']

# Assign the continent in each respective continent dataframe to each country in io_codes as an additional column
io_codes['continent'] = None                                                                # Create a new column for continents

for idx, row in io_codes.iterrows():
    country = row['area']  # Get the country name from the 'area' column

    # Check if the country is in each continent dataframe and assign the continent accordingly
    if country in east_southeast_asia['area'].values:
        io_codes.at[idx, 'continent'] = 'East Asia and Southeast Asia'
    elif country in east_europe['area'].values:
        io_codes.at[idx, 'continent'] = 'Eastern Europe'
    elif country in latam_caribbean['area'].values:
        io_codes.at[idx, 'continent'] = 'Latin America and the Caribbean'
    elif country in east_north_africa['area'].values:
        io_codes.at[idx, 'continent'] = 'Near East and North Africa'
    elif country in north_america['area'].values:
        io_codes.at[idx, 'continent'] = 'North America'
    elif country in oceania['area'].values:
        io_codes.at[idx, 'continent'] = 'Oceania'
    elif country in rest_of_world['area'].values:
        io_codes.at[idx, 'continent'] = 'RoW'
    elif country in russian_fed['area'].values:
        io_codes.at[idx, 'continent'] = 'Russian Federation'        
    elif country in south_asia['area'].values:
        io_codes.at[idx, 'continent'] = 'South Asia'        
    elif country in sub_saharan_africa['area'].values:
        io_codes.at[idx, 'continent'] = 'Sub-Saharan Africa'        
    elif country in west_europe['area'].values:
        io_codes.at[idx, 'continent'] = 'Western Europe'        

# Order the labels
order_io = ['area_code', 'continent', 'area', 'item_code', 'item', 'comm_code', 'comm_group', 'group']
io_codes = io_codes[order_io]

# Repeat the above for fd_codes too to add for each of the rows what the corresponding continent is as an additional column
# Assign the continent in fd_codes dataframe to each country in fd_codes as an additional column
fd_codes['continent'] = None                                                                # Initialise a new column for continents

# Iterate through each row in fd_codes
for idx, row in fd_codes.iterrows():
    country = row['area']  # Get the country name from the 'area' column

    # Check if the country is in each continent dataframe and assign the continent accordingly
    if country in east_southeast_asia['area'].values:
        fd_codes.at[idx, 'continent'] = 'East Asia and Southeast Asia'
    elif country in east_europe['area'].values:
        fd_codes.at[idx, 'continent'] = 'Eastern Europe'
    elif country in latam_caribbean['area'].values:
        fd_codes.at[idx, 'continent'] = 'Latin America and the Caribbean'
    elif country in east_north_africa['area'].values:
        fd_codes.at[idx, 'continent'] = 'Near East and North Africa'
    elif country in north_america['area'].values:
        fd_codes.at[idx, 'continent'] = 'North America'
    elif country in oceania['area'].values:
        fd_codes.at[idx, 'continent'] = 'Oceania'
    elif country in rest_of_world['area'].values:
        fd_codes.at[idx, 'continent'] = 'RoW'
    elif country in russian_fed['area'].values:
        fd_codes.at[idx, 'continent'] = 'Russian Federation'        
    elif country in south_asia['area'].values:
        fd_codes.at[idx, 'continent'] = 'South Asia'        
    elif country in sub_saharan_africa['area'].values:
        fd_codes.at[idx, 'continent'] = 'Sub-Saharan Africa'        
    elif country in west_europe['area'].values:
        fd_codes.at[idx, 'continent'] = 'Western Europe'        


# Order the labels
order_fd = ['area_code', 'continent', 'area', 'fd']
order_fd = fd_codes[order_fd]

#%% For Y: Applying labels and exporting ungrouped processed data to excel

# Make copy of Y
Y_ungrouped = Y.copy('deep')

# Create DataFrame for column labels, combining 'area' and 'fd' columns from fd_codes
Y_labels_cols_df = fd_codes[['area', 'fd', 'continent']].copy()

# Attribute labels to Y and export ungrouped version to excel
Y_labels_rows = io_codes.drop(labels=["area_code", "item_code", "comm_code", "comm_group", "group"], axis=1)            # Drop the specified columns from io_codes to get only relevant data (continent and area) for row labels in later used Y_grouped
Y_labels_rows_df = pd.DataFrame(Y_labels_rows)  # Create DataFrame for row labels

Y_ungrouped.columns = pd.MultiIndex.from_frame(Y_labels_cols_df)                                                        # Assign the MultiIndex for columns using the column labels DataFrame
Y_ungrouped.index = pd.MultiIndex.from_frame(Y_labels_rows_df)                                                          # Repeat for rows

output_path_Y_ungrouped = file_path + r"\Y_ungrouped.csv"                                                               # Define the output file path where the Excel will be saved
Y_ungrouped.to_csv(output_path_Y_ungrouped, index=True)                                                                 # Export Y_grouped to an Excel file, specifying the sheet name


#%% For x: Applying labels, grouping by continents, and exporting to excel
x_ungrouped = x.copy('deep')
    
x_labels_rows_df = Y_labels_rows_df.copy()

x_ungrouped.index = pd.MultiIndex.from_frame(x_labels_rows_df)                                                          # Repeat for rows

# Save the processed x_ungrouped to an Excel file
output_path_x_ungrouped = file_path + r"\x_ungrouped.csv"
x_ungrouped.to_csv(output_path_x_ungrouped, index=True)


#%% For Z: Applying labels, grouping by continents, and exporting to excel
Z_ungrouped = Z.copy('deep')

# Assign labels to the rows
Z_labels_rows_df = Y_labels_rows_df.copy()
Z_ungrouped.index = pd.MultiIndex.from_frame(Z_labels_rows_df)                                                          # Repeat for rows

# Assign these same labels to the columns
Z_labels_cols_df = Y_labels_rows_df.copy()
Z_ungrouped.columns = pd.MultiIndex.from_frame(Z_labels_cols_df)

# Save the processed x_ungrouped to a CSV file (too big for excel)
output_path_Z_ungrouped = file_path + r"\Z_ungrouped.csv"
Z_ungrouped.to_csv(output_path_Z_ungrouped, index=True)

# Print to test the row and column labels
print("First 20 row labels of Z_ungrouped:")
print(Z_ungrouped.head(20).index)
print("\nFirst 20 column labels of Z_ungrouped:")
print(Z_ungrouped.columns[:20])

#%%
# The other extensions are already in an easy to use format, so we only clean up the land use df a bit here
# For land use F: import into dataframe, add labels for country and continent, then export to excel for later results
F_ungrouped = land_use.copy()

# Assign labels to the rows
F_labels_rows_df = Y_labels_rows_df.copy()
F_ungrouped.index = pd.MultiIndex.from_frame(F_labels_rows_df)   

# Add in additional relevant columns to the index
F_ungrouped = F_ungrouped.set_index(['iso3c', 'comm_code'], append=True)

# Save the processed F_ungrouped to a CSV file
output_path_F_ungrouped = file_path + r"\F_ungrouped.csv"
F_ungrouped.to_csv(output_path_F_ungrouped, index=True)


#%%
# Export labels from Y for the rows to be used for labelling L in other parts of the model
output_path_L_export = file_path + r"\labels_continent_country_item.csv"                                                                  
Y_labels_rows_df.to_csv(output_path_L_export, index=True)











