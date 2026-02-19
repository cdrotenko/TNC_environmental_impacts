"""
Created on Thu Jan 16 11:09:50 2025 by Christina Drotenko

Purpose of code: modelling
Input: processed data
Output: results

"""

#%% Import all the required data from 3 - aggregation later in process.py

# Imports
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, diags
import os
from openpyxl import Workbook

#%%

#CD TO DO: create placeholders instead in the file names below

# Read data and applying indices

# ADD IN YOUR RAW DATA FILE PATH, PLACE TO STORE RESULTS, PROCESSED DATA BELOW FILE PATHS BELOW
#file_path = r"INSERTPATHHERE\Supplementary Data\data\1 - raw data"
#file_path_processed_data = r"INSERTPATHHERE\Supplementary Data\data\2 - processed data"
#file_path_results = r"INSERTPATHHERE\Supplementary Data\data\3 - results"

x_import = pd.read_csv(file_path + r"\x_ungrouped.csv", delimiter=',', index_col=[0, 1, 2])
Y_import = pd.read_csv(file_path + r"\Y_ungrouped.csv", delimiter=',', index_col=[0, 1, 2], header=[0, 1, 2], skiprows=[3])
Z_import = pd.read_csv(file_path + r"\Z_ungrouped.csv", delimiter=',', index_col=[0, 1, 2], header=[0, 1, 2], skiprows=[3])
f_import = pd.read_csv(file_path + r"\F_ungrouped.csv", delimiter=',', index_col=[0, 1, 2, 3, 4])

# Add labels to df for use later after calculations
x_cols = x_import.index.to_frame(index=None)
Y_cols = Y_import.index.to_frame(index=None)
Z_cols = Z_import.index.to_frame(index=None)
f_cols = f_import.index.to_frame(index=None)

# Make copies of the variables for the calculations
x = x_import.copy()
Y = Y_import.copy()
Z = Z_import.copy()
f_land_use = f_import.copy()

# Calculate x myself to ensure there are no bugs in the imported version
x_calc = Z.sum(axis = 1) + Y.sum(axis = 1)

# After checking the differences with x, then can make x be x_calc.copy()
x = x_calc.copy() 

# Read in Profundo data that has had additional processing inside excel
# Read Cattle, Chicken, Pigs data for company / continent pairings (aggregation by continent is already in the Excel) - Dairy is already split by weight
bovine_meat = pd.read_excel(file_path_processed_data + r"\TNC production by weight.xlsx", sheet_name="Cattle weight").iloc[:, :-2]         # unit = tonnes
poultry_meat = pd.read_excel(file_path_processed_data + r"\TNC production by weight.xlsx", sheet_name="Chickens weight").iloc[:, :-2]      # unit = tonnes
pig_meat = pd.read_excel(file_path_processed_data + r"\TNC production by weight.xlsx", sheet_name="Pigs weight").iloc[:, :-2]              # unit = tonnes

# Read in dairy data which is already in weight (no need to convert)
dairy = pd.read_excel(file_path_processed_data + r"\TNC production by weight.xlsx", sheet_name="Dairy")                       # unit = tonnes

# indices
bovine_meat.set_index([bovine_meat.columns[0], bovine_meat.columns[1]], inplace = True)
poultry_meat.set_index([poultry_meat.columns[0], poultry_meat.columns[1]], inplace = True)
pig_meat.set_index([pig_meat.columns[0], pig_meat.columns[1]], inplace = True)
dairy.set_index([dairy.columns[0], dairy.columns[1]], inplace = True)

# renaming these to match the original variables for less renaming later in the code
cattle = bovine_meat.copy()
chickens = poultry_meat.copy()
pigs = pig_meat.copy()

# Store f index for use later
f_index = f_import.copy()

# Create empty excel file for results to be stored in
# Define paths
file_results_storage = file_path_results
file_name_results = "Results_all_impacts.xlsx"

# Combine
full_path = os.path.join(file_results_storage, file_name_results)

# Create and save workbook
results_excel_file = Workbook()
results_excel_file.save(full_path)


#%%
# Adding in the extensions beyond LU - although no additional processing has been conducted on these files, they are in the 2 - processed data folder for simplicity
f_blue_water = pd.read_csv(file_path_processed_data + r"\blue_water.csv", delimiter=',', index_col=[0, 1, 2, 3])
f_green_water = pd.read_csv(file_path_processed_data + r"\green_water.csv", delimiter=',', index_col=[0, 1, 2, 3])
f_n_application = pd.read_csv(file_path_processed_data + r"\N_application.csv", delimiter=',', index_col=[0, 1, 2, 3])

# Flip the order of these df indices to match f_land_use
f_blue_water = f_blue_water.swaplevel(0, 1)
f_green_water = f_green_water.swaplevel(0, 1)
f_n_application = f_n_application.swaplevel(0, 1)


#%% Footprint calculation up to L + deleting A, I for memory
x_new = np.array(x)
xinv = ((x_new !=0) / (x_new + (x_new ==0)))

A = Z @ np.diagflat(xinv)

I = np.identity(A.shape[0])

L = np.linalg.inv(I - A)

del A, I



#%% Choice for animal product (Ctrl+1 to select / deselect) - and then if you would like all of them, run one after the other with the full code following it

# Make ONE choice of the following to become the animal product that we calculate for

# # 1 - Bovine meat
# animal_product = "Bovine Meat"
# file_name_animal = "Bovine_meat"
# animal_name = cattle.copy()
# animal_weight = "Cattle weight"
# graph_label_animal = "Bovine meat "

# # 2 - Poultry meat
# animal_product = "Poultry Meat"
# file_name_animal = "Poultry_meat"
# animal_name = chickens.copy()
# animal_weight = "Chickens weight"
# graph_label_animal = "Poultry meat "


# # 3 - Pigmeat
# animal_product = "Pigmeat"
# file_name_animal = "Pigmeat"
# animal_name = pigs.copy()
# animal_weight = "Pigs weight"
# graph_label_animal = "Pigmeat "


# 4 - Dairy - already in weight so no animal_weight variable
animal_product = "Milk - Excluding Butter"
file_name_animal = "Milk - Excluding Butter"
animal_name = dairy.copy()
animal_weight = "Dairy"
graph_label_animal = "Dairy "

## always run the following:
# Create dictionary for all the conditions of footprints
footprint_types = [
    {
        "data": f_land_use,
        "extension_name": "Land Use Footprint (ha)",
        "title": f"Total Land Use Footprints by Company for {animal_product} (ha)",
        "filename": "all_lu_footprints_grouped.jpg",
        "title_div1k": f"Total Land Use Footprints by Company for {animal_product} (kha)",
        "filename_div1k": f"{file_name_animal}_lu_footprints.jpg",
        "excel_export": f"{file_name_animal}_lu_footprints",
        "graph_label_fp": "land use ",
        "graph_unit_top": "(Mha)",
        "graph_unit_bottom": "(kha)",
        "df_label": "Land use"
    },
    {
        "data": f_blue_water,
        "extension_name": "Blue Water Footprint (m³)",
        "title": f"Total Blue Water Footprints by Company for {animal_product} (m³)",
        "filename": "all_bw_footprints_grouped.jpg",
        "title_div1k": f"Total Blue Water Footprints by Company for {animal_product} (1000 m³)",
        "filename_div1k": f"{file_name_animal}_bw_footprints.jpg",
        "excel_export": f"{file_name_animal}_bw_footprints",
        "graph_label_fp": "blue water use ",
        "graph_unit_top": "(1,000,000 m³)",
        "graph_unit_bottom": "(1,000 m³)",
        "df_label": "Blue water"
    },
    {
        "data": f_green_water,
        "extension_name": "Green Water Footprint (m³)",
        "title": f"Total Green Water Footprints by Company for {animal_product} (m³)",
        "filename": "all_gw_footprints_grouped.jpg",
        "title_div1k": f"Total Green Water Footprints by Company for {animal_product} (1000 m³)",
        "filename_div1k": f"{file_name_animal}_gw_footprints.jpg",
        "excel_export": f"{file_name_animal}_gw_footprints",
        "graph_label_fp": "green water use ",
        "graph_unit_top": "(1,000,000 m³)",
        "graph_unit_bottom": "(1,000 m³)",
        "df_label": "Green water"
    },
    {
        "data": f_n_application,
        "extension_name": "Nitrogen Application (kg)",
        "title": f"Total Nitrogen Application by Company for {animal_product} (kg)",
        "filename": "all_n_app_footprints_grouped.jpg",
        "title_div1k": f"Total Nitrogen Application by Company for {animal_product} (t)",
        "filename_div1k": f"{file_name_animal}_n_app.jpg",
        "excel_export": f"{file_name_animal}_n_application",
        "graph_label_fp": "nitrogen application ",
        "graph_unit_top": "(kt)",
        "graph_unit_bottom": "(t)",
        "df_label": "Nitrogen application"
    }

]

#%% FOR TOTALS CALCS - everything excepting including company data
# Create empty DataFrame to store all results
all_footprints_df = pd.DataFrame()

# Keep it separate from the next part so that it doesn't overwrite the df

#%% This part calculates the environmental impacts that are in FABIO for comparison in the paper to global env impacts
# Then re-run this 4x for each desired animal product or of course just create a loop if you prefer that

# Loop through each footprint type
for fp in footprint_types:
    f_total = fp["data"].copy()
    title = fp["title"]
    
    # Flatten f_total and multiply by xinv to get intensity B_total
    B_total = np.array(f_total).flatten() @ np.diagflat(xinv)
    
    # Prep Y for this animal product
    # Collapse Y into one summed column (global consumption baseline)
    Y_new = Y.sum(axis=1)
    
    # Ensure Y_new is a DataFrame with a column called 'Values'
    Y_new = Y_new.to_frame(name='Values')
    Y_new = Y_new.rename_axis(['continent', 'country', 'item'])
    
    # Template of zeros
    Y_animal = Y_new.copy()
    Y_animal['Values'] = 0.0
    
    # Keep only the animal product of interest
    mask = Y_animal.index.get_level_values("item") == animal_product
    Y_animal.loc[mask, 'Values'] = Y_new.loc[mask, 'Values']
    
    # Reset index to ensure compatibility for multiplication
    Y_animal = Y_animal.reset_index().set_index(['continent', 'country', 'item'])
    
    # Calculate footprint
    footprint_animal = B_total @ L @ Y_animal
    print (footprint_animal)
    print (animal_product)
    print (fp['df_label'])
    
    # Add a new row to the df
    all_footprints_df = pd.concat([
        all_footprints_df,
        pd.DataFrame([{
            'Footprint': footprint_animal.sum(),
            'Animal_product': animal_product,
            'Footprint_type': fp['extension_name']
        }])
    ], ignore_index=True)
    

#%% Below is the analysis of the TNC impacts in a big for loop

# Empty list for all regions & impacts for exporting
all_regions_list = []

# Loop through each footprint type
for fp in footprint_types:
    f_total = fp["data"].copy()
    extension_name = fp["extension_name"]
    title = fp["title"]
    filename = fp["filename"]
    title_div1k = fp["title_div1k"]
    filename_div1k = fp["filename_div1k"]
    excel_export = fp["excel_export"]
    graph_label_fp = fp["graph_label_fp"]
    graph_unit_top = fp["graph_unit_top"]
    graph_unit_bottom = fp["graph_unit_bottom"]
    df_label = fp["df_label"]



    # Footprint calculation after L - for land use
    #!! need to have this be on a per unit of output so need to go from E to F (go from absolute land use to land use intensity) --> environmental flows to coefficients
    B_total = np.array(f_total).flatten() @ np.diagflat(xinv)
    
    # Footprint total
    footprint_grouped = B_total @ L @ Y
    
    # Footprint total summed
    footprint_grouped_sum = pd.array(footprint_grouped).sum(axis=0)
    
    #--next-#
    # Footprint total food columns only
    Y_food = Y.loc[:, Y.columns.get_level_values(1) == 'food']
    footprint_grouped_food = B_total @ L @ Y_food
    
    # Footprint total summed food columns only
    footprint_grouped_sum_food = pd.array(footprint_grouped_food).sum(axis=0)
    
    
    # Sum the Y food columns into one and create a relative FABIO dataframe from Z to figure out the percentages that each country makes up of each continent
    # This relative dataframe is based on the intermediate products Z and not the end Y products --> and put into the df Y_percentages_FABIO_relative
    
    #!! Create relative df based on X for dairy (but add it into Y_agg since it is applied to Y later)
    Y_agg = x.copy('deep')
    Y_agg.name = "Values"                               # Rename the column with the data to be called Values
    Y_agg = Y_agg.rename_axis(['continent', 'country', 'item'])
    
    # Filter the rows to be only one item since this is the % of eg EUR NL Milk - Excluding Butter from Milk - Excluding Butter, and EUR DE Milk - Excluding Butter from EUR Milk - Excluding Butter
    Y_agg = Y_agg.loc[Y_agg.index.get_level_values(2) == animal_product, :]
    
    # Copy the data
    Y_agg_proportions = Y_agg.copy()
    
    # Get row continent labels
    row_continents = Y_agg_proportions.index.get_level_values(0)
    
    # Ensure Y_normalized is a series not a df
    Y_normalized = pd.Series(index=Y_agg_proportions.index, dtype=float)
    
    # Repeat with unique continents
    for continent in row_continents.unique():
        # Find rows and columns for the given continent
        row_mask = row_continents == continent
    
        # Get the sum for this continent
        col_sum = Y_agg_proportions.loc[row_mask].sum()
    
        # Normalise only these rows
        Y_normalized.loc[row_mask] = Y_agg_proportions.loc[row_mask] / col_sum
    
    
    # Convert to numeric
    Y_normalized = Y_normalized.astype(float)
    
    # Check column sums again to make sure they = 1
    column_sums = Y_normalized.sum(axis=0)
    print(column_sums)
    
    
    # Adding company data into Y
    
    # Collapse Y into one summed column (global consumption baseline)
    Y_new = Y.sum(axis=1)
    Y_new.name = "Values"
    Y_new = Y_new.rename_axis(['continent', 'country', 'item'])
    
    # Start a template of zeros (same shape as Y_new)
    Y_zeros = Y_new.copy()
    Y_zeros[:] = 0
    
    # Dictionary to hold company-specific Y matrices
    Y_dict_animal = {}
    
    # Loop over companies
    for company in animal_name.index.get_level_values(0).unique():
        # Start from zero template
        Y_company = Y_zeros.copy()
    
        # Extract this company's animal demand (indexed by Region)
        filtered_df = animal_name.loc[company]
    
        # Fill in values for the relevant (continent, animal product) rows
        for region, value in filtered_df[animal_weight].items():
            if (region, animal_product) in Y_company.index.droplevel("country"):
                # Repeat across all levels
                mask = (
                    (Y_company.index.get_level_values("continent") == region) &
                    (Y_company.index.get_level_values("item") == animal_product)
                )
                Y_company.loc[mask] = value
    
        # Store
        Y_dict_animal[company] = Y_company
    
    
    # Now in each of the Y dfs add the country column into the index and put it in the same order as Z_normalised for multiplication later
    Y_dict_animal_weighted = Y_dict_animal.copy()
    
    for key, df in Y_dict_animal.items():
        Y_dict_animal_weighted[key] = df.reset_index().set_index(["continent", "country", "item"])        # Reorder index and add country to the index
    
    # copy
    relative_Y = Y_normalized.copy()
    
    # Remove the index to replace the item with pigs and then reset it within the index again
    relative_Y = relative_Y.reset_index()
    relative_Y.iloc[:, 2] = animal_product
    relative_Y.set_index([relative_Y.columns[0], relative_Y.columns[1], relative_Y.columns[2]], inplace = True)
    relative_Y = relative_Y.rename_axis(['continent', 'country', 'item'])
    relative_Y.columns = ['Proportion']  # Rename the column to "Values"
    
    ### Then do the matching
    # Disaggregate company Ys: Multiply each of the Y company dataframes by the corresponding Z_normalized values to attribute the %s to each
    Y_dict_animal_disagg = Y_dict_animal_weighted.copy()
    
    # Loop over dictionary of dataframes
    for key, Y_df in Y_dict_animal_disagg.items():
        # Ensure indices match in terms of length and order
        if not (Y_df.index.equals(relative_Y.index)):
            # Align Y_df's index with relative_Z's index
            Y_df = Y_df.reindex(relative_Y.index)
        
        # Multiply 'Values' column in Y_df by 'Proportion' column in relative_Z
        Y_dict_animal_disagg[key]['Values'] *= relative_Y['Proportion']
    
    # Replace NaNs with 0s (do this outside of the loop otherwise it makes the totals readjust)
    for key, Y_df in Y_dict_animal_disagg.items():
        Y_dict_animal_disagg[key]['Values'] = Y_dict_animal_disagg[key]['Values'].fillna(0)
        
    # Quick test for the sums of Y_dict_animal_disagg companies to make sure they match the original Y_dict_animal totals
    for key, Y_df in Y_dict_animal_disagg.items():
        sum_values = Y_df['Values'].sum()
        print(f"Sum of 'Values' for {key}: {sum_values}")
        
    
    
    # Running code above in a way where it shows where the impacts occur (hotspot analysis)
    
    
    # Convert to sparse once outside the loop
    L_sparse = csr_matrix(L)
    B_diag_sparse = diags(B_total)   # sparse diagonal matrix
    
    F_dict_animal = {}
    
    # Loop over the companies
    sample_companies = list(Y_dict_animal_disagg.items())
    
    for first_company_name, first_company_Y in sample_companies:
        cleaned_name = first_company_name.replace(' ', '_')
    
        # Convert Y to sparse vector
        Y_vec_sparse = csr_matrix(first_company_Y.sum(axis=1).values).T
    
        # Compute footprint_new using sparse formula (same as before)
        footprint_new = B_diag_sparse @ L_sparse @ Y_vec_sparse
    
        # Convert back to dense array
        footprint_new = np.asarray(footprint_new.todense()).flatten()
    
        # Build df
        F_country = pd.DataFrame(
            footprint_new, columns=['land_use_footprint'], index=f_import.index
        )
    
        # Grazing footprint - aggregated
        F_grazing = F_country.loc[F_country.index.get_level_values(2) == 'Grazing'] \
                              .groupby(level=0).sum()
    
        # Nongrazing footprint - aggregated
        F_nongrazing = F_country.loc[F_country.index.get_level_values(2) != 'Grazing'] \
                                 .groupby(level=0).sum()
    
        # Combine grazing and nongrazing
        combined_footprints = pd.concat([F_grazing, F_nongrazing], axis=1)
        combined_footprints.columns = ['Grazing', 'Nongrazing']
        combined_footprints = combined_footprints.where(combined_footprints >= 1, 0)
    
        # Store in dictionary
        F_dict_animal[cleaned_name] = {
            'footprint_grazing': F_grazing,
            'footprint_nongrazing': F_nongrazing,
            'combined_footprints': combined_footprints
        }

    # Company impacts - store them in a way so that these are sorted clearly in the excel export
    # Create empty lists to store DataFrames - there are options for both rounded & unrounded depending on what is needed
    all_footprints_list_unrounded = []
    all_footprints_list_rounded = []
    
    # Loop through each company and process footprints
    for company_name, footprints_data in F_dict_animal.items():
        # Extract footprints for grazing and croplands
        F_grazing = footprints_data['footprint_grazing'].copy()
        F_nongrazing = footprints_data['footprint_nongrazing'].copy()
    
        # Div by 1000 conversion of units
        F_grazing_div1k = F_grazing / 1000
        F_nongrazing_div1k = F_nongrazing / 1000
    
        # Create copies for both versions
        F_grazing_unrounded = F_grazing_div1k.copy()
        F_nongrazing_unrounded = F_nongrazing_div1k.copy()
    
        F_grazing_rounded = F_grazing_div1k.where(F_grazing_div1k >= 1, 0).round(2)
        F_nongrazing_rounded = F_nongrazing_div1k.where(F_nongrazing_div1k >= 1, 0).round(2)
    
        # Add company and land type columns
        for df, land_type in [(F_grazing_unrounded, "Grazing"), (F_nongrazing_unrounded, "Croplands")]:
            df["Company"] = company_name
            df["Land_Type"] = land_type
            all_footprints_list_unrounded.append(df)
    
        for df, land_type in [(F_grazing_rounded, "Grazing"), (F_nongrazing_rounded, "Croplands")]:
            df["Company"] = company_name
            df["Land_Type"] = land_type
            all_footprints_list_rounded.append(df)
    
    # Combine all companies into single DataFrames
    all_footprints_df_unrounded = pd.concat(all_footprints_list_unrounded).reset_index()
    all_footprints_df_rounded = pd.concat(all_footprints_list_rounded).reset_index()
    
    # Reorder columns for readability
    cols = ["Company", "Land_Type"] + [col for col in all_footprints_df_unrounded.columns if col not in ["Company", "Land_Type"]]
    all_footprints_df_unrounded = all_footprints_df_unrounded[cols]
    all_footprints_df_rounded = all_footprints_df_rounded[cols]
    
    
    # Prepare the company regional footprints dataframe for exporting
    df_regions = all_footprints_df_rounded.copy()
    
    # Ensure there's acolumn called 'Animal'
    if 'Animal' not in df_regions.columns:
        df_regions['Animal'] = animal_product  # current animal_product in loop
    
    # Add a column for the impact type
    if 'Footprint' not in df_regions.columns:
        df_regions['Footprint'] = df_label

    # Add a column for the unit type
    if 'Unit' not in df_regions.columns:
        df_regions['Unit'] = graph_unit_bottom
 
    
    # Detect numeric footprint column
    num_cols = df_regions.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        if 'land_use_footprint' in df_regions.columns:
            value_col = 'land_use_footprint'
        else:
            raise ValueError("Couldn't find numeric footprint column beep beep error.")
    else:
        value_col = 'land_use_footprint' if 'land_use_footprint' in num_cols else num_cols[0]
    
    # Remove underlines in the company names before exporting
    df_regions["Company"] = df_regions["Company"].str.replace("_", " ", regex=False)
    
    # Add this df to the full list
    all_regions_list.append(df_regions)

# Combine all & rename column
all_regions_df = pd.concat(all_regions_list, ignore_index=True)

all_regions_df = all_regions_df.rename(columns = {
    "land_use_footprint":"Footprint amount",
    "continent":"Region",
    "Animal":"Animal product",
    "Land_Type":"Land type"})

# Round
all_regions_df["Footprint amount"] = all_regions_df["Footprint amount"].round(0)
    
# Export results to excel
with pd.ExcelWriter(full_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    all_regions_df.to_excel(writer, sheet_name=animal_product, index=False)
    
