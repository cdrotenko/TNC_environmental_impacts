# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:09:50 2025

@author: christina

Purpose of code: model
Input: processed data
Output: results

"""

#%% Import all the required data from 3 - aggregation later in process.py

# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags
from matplotlib.ticker import FuncFormatter
import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import zipfile, os
from openpyxl import Workbook

#%%

# Read data and applying indices
file_path = r"C:\Users\chris\OneDrive - Universiteit Leiden\Documents\Data\FABIO_old\FABIO_old"
file_path_results = r"C:\Users\chris\OneDrive - Universiteit Leiden\Documents\Data\FABIO_old\FABIO_old\Results\\"

x_import = pd.read_csv(file_path + r"\x_ungrouped.csv", delimiter=',', index_col=[0, 1, 2])
Y_import = pd.read_csv(file_path + r"\Y_ungrouped.csv", delimiter=',', index_col=[0, 1, 2], header=[0, 1, 2], skiprows=[3])
Z_import = pd.read_csv(file_path + r"\Z_ungrouped.csv", delimiter=',', index_col=[0, 1, 2], header=[0, 1, 2], skiprows=[3])
f_import = pd.read_csv(file_path + r"\F_ungrouped.csv", delimiter=',', index_col=[0, 1, 2, 3, 4])
#n_import = pd.read_csv(file_path + r"\N_balance_with_leaching.csv", delimiter=',')

# Add labels to df for use later after calculations
#NOTE didn't do this for the index of the rows (not sure if there's a better way than just resetting the index eg of x and then putting the first 3 rows into a separate col --> how to more efficiently 1) save an index into a df from a df and then later reapply it?)
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

# Read in Profundo data
# Read Cattle, Chicken, Pigs data for company / continent pairings (aggregation by continent is already in the Excel) - Dairy is already split by weight
bovine_meat = pd.read_excel(file_path + r"\Profundo data.xlsx", sheet_name="Cattle weight").iloc[:, :-2]         # unit = tonnes
poultry_meat = pd.read_excel(file_path + r"\Profundo data.xlsx", sheet_name="Chickens weight").iloc[:, :-2]      # unit = tonnes
pig_meat = pd.read_excel(file_path + r"\Profundo data.xlsx", sheet_name="Pigs weight").iloc[:, :-2]              # unit = tonnes

# Read in dairy data which is already in weight (no need to convert)
dairy = pd.read_excel(file_path + r"\Profundo data.xlsx", sheet_name="Dairy")                       # unit = tonnes

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
file_results_storage = r"C:\Users\chris\OneDrive - Universiteit Leiden\Documents\Data\FABIO_old\FABIO_old"
file_name_results = "Results_all_impacts.xlsx"

# Combine
full_path = os.path.join(file_results_storage, file_name_results)

# Ensure the folder exists
os.makedirs(os.path.dirname(full_path), exist_ok=True)

# Create and save workbook
results_excel_file = Workbook()
results_excel_file.save(full_path)

print(f"Saved Excel file to: {full_path}")


#%%
# Adding in the extensions beyond LU from Jiahui
f_blue_water = pd.read_csv(file_path + r"\blue_water.csv", delimiter=',', index_col=[0, 1, 2, 3])
f_green_water = pd.read_csv(file_path + r"\green_water.csv", delimiter=',', index_col=[0, 1, 2, 3])
f_n_application = pd.read_csv(file_path + r"\N_application.csv", delimiter=',', index_col=[0, 1, 2, 3])
f_p_application = pd.read_csv(file_path + r"\P_application.csv", delimiter=',', index_col=[0, 1, 2, 3])

# Flip the order of these df indices to match f_land_use
f_blue_water = f_blue_water.swaplevel(0, 1)
f_green_water = f_green_water.swaplevel(0, 1)
f_n_application = f_n_application.swaplevel(0, 1)
f_p_application = f_p_application.swaplevel(0, 1)


#%% Footprint calculation up to L - with aggregation
x_new = np.array(x)
xinv = ((x_new !=0) / (x_new + (x_new ==0)))

A = Z @ np.diagflat(xinv)

I = np.identity(A.shape[0])

L = np.linalg.inv(I - A)

del A, I



#%% Choice for animal product (Ctrl+1 to select / deselect)

# Make ONE choice of the following to become the animal we calculate for

# # Bovine meat
# animal_product = "Bovine Meat"
# file_name_animal = "Bovine_meat"
# animal_name = cattle.copy()
# animal_weight = "Cattle weight"
# graph_label_animal = "Bovine meat "

# # Poultry meat
# animal_product = "Poultry Meat"
# file_name_animal = "Poultry_meat"
# animal_name = chickens.copy()
# animal_weight = "Chickens weight"
# graph_label_animal = "Poultry meat "


# # Pigmeat
# animal_product = "Pigmeat"
# file_name_animal = "Pigmeat"
# animal_name = pigs.copy()
# animal_weight = "Pigs weight"
# graph_label_animal = "Pigmeat "


# Dairy - already in weight so no animal_weight variable
animal_product = "Milk - Excluding Butter"
file_name_animal = "Milk - Excluding Butter"
animal_name = dairy.copy()
animal_weight = "Dairy"
graph_label_animal = "Dairy "


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
    },
    {
        "data": f_p_application,
        "extension_name": "Phosphorus Application (kg)",
        "title": f"Total Phosphorus Application by Company for {animal_product} (kg)",
        "filename": "all_p_app_footprints_grouped.jpg",
        "title_div1k": f"Total Phosphorus Application by Company for {animal_product} (t)",
        "filename_div1k": f"{file_name_animal}_p_app.jpg",
        "excel_export": f"{file_name_animal}_p_application",
        "graph_label_fp": "phosphorus application ",
        "graph_unit_top": "(kt)",
        "graph_unit_bottom": "(t)",
        "df_label": "Phosphorus application"
    }
]

#%% FOR TOTALS CALCS - everything excepting including company data
# Create empty DataFrame to store all results
all_footprints_df = pd.DataFrame()

# Keep it separate so that it doesn't overwrite the df

#%%

# Loop through each footprint type
for fp in footprint_types:
    f_total = fp["data"].copy()
    title = fp["title"]
    
    # Flatten f_total and multiply by xinv to get intensity B_total
    B_total = np.array(f_total).flatten() @ np.diagflat(xinv)
    
    # Collapse Y into one summed column
    Y_new = Y.sum(axis=1)
    
    # rename col
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
    
    # Add a new row to the DataFrame
    all_footprints_df = pd.concat([
        all_footprints_df,
        pd.DataFrame([{
            'Footprint': footprint_animal.sum(),
            'Animal_product': animal_product,
            'Footprint_type': fp['extension_name']
        }])
    ], ignore_index=True)
    

#%% ABOVE PART IS FOR GRAPHS, BELOW IS FOR THE ACTUAL ANALYSIS





#%%
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
    
    # Ensure Y_normalized is a series, not a df
    Y_normalized = pd.Series(index=Y_agg_proportions.index, dtype=float)
    
    # Iterate over unique continents
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
                # Because Y_company index has 3 levels, we need to broadcast over all countries
                mask = (
                    (Y_company.index.get_level_values("continent") == region) &
                    (Y_company.index.get_level_values("item") == animal_product)
                )
                Y_company.loc[mask] = value
    
        # Store result
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
    
        # Build DataFrame
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
    
    
    # Now add it into a df for the graphs
    # Create an empty list to store the rows for all_footprints_grouped
    all_footprints_rows = []
    
    # Loop through each company in F_dict_animal
    for company_name in sorted(F_dict_animal.keys()):
        footprints_data = F_dict_animal[company_name]
        # Extract the combined_footprints DataFrame for the current company
        combined_footprints = footprints_data['combined_footprints']
    
        # Sum the Grazing and Nongrazing columns
        grazing_sum = combined_footprints['Grazing'].sum()
        nongrazing_sum = combined_footprints['Nongrazing'].sum()
    
        # Create a row with the summed values and the company name as the index
        company_row = pd.Series([grazing_sum, nongrazing_sum], index=['Grazing', 'Nongrazing'], name=company_name)
    
        # Append the row to the list
        all_footprints_rows.append(company_row)
    
    # Convert the list of rows into a DataFrame
    all_footprints_grouped = pd.DataFrame(all_footprints_rows)
    
    # Sort the DataFrame by the index (company names) in alphabetical order
    all_footprints_grouped = all_footprints_grouped.reindex(sorted(all_footprints_grouped.index))    

    # Rename 'Nongrazing' to 'Croplands' for improved understanding
    all_footprints_grouped = all_footprints_grouped.rename(columns={'Nongrazing': 'Croplands'})
    
    # Print the resulting DataFrame to verify
    print(all_footprints_grouped)
        
    # Make the tables
    
    def format_numbers_with_commas(x, pos):
        """Formatter function to add commas to large numbers."""
        return f'{x:,.0f}'
    
    def save_heatmap_as_image(df, filename, title):
        """Save a heatmap with totals displayed outside, below the figure."""
        # Compute totals separately (without modifying df)
        total_grazing = df["Grazing"].sum()
        total_croplands = df["Croplands"].sum()
    
        # Create a figure for the heatmap
        fig, ax = plt.subplots(figsize=(10, len(df) * 0.3 + 1))  # Extra space for text
    
        # Create the heatmap
        heatmap = sns.heatmap(df, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Footprint'}, fmt=".0f", linewidths=0.5, ax=ax)
    
        # Format the numbers in the heatmap cells with commas
        for text in heatmap.texts:
            text.set_text(f'{int(float(text.get_text())):,.0f}')  # Format numbers with commas
    
        # Format the colorbar with commas
        heatmap.collections[0].colorbar.ax.yaxis.set_major_formatter(FuncFormatter(format_numbers_with_commas))
    
        # Add a title above the heatmap
        plt.title(title, fontsize=14, fontweight='bold')
    
        # Add totals below the heatmap
        plt.figtext(0.5, -0.08, 
                    f"Total Grazing: {total_grazing:,.0f} |  Total Croplands: {total_croplands:,.0f} ", 
                    ha="center", fontsize=12, fontweight='bold')
    
        # Concatenate the file path and filename for the final save path
        file_path = file_path_results + filename  # Ensure this is the full directory path
    
        # Save the heatmap as a JPG in the specified directory
        print(f"Saving image to: {file_path}")  # Debug: Check the path before saving
        plt.tight_layout()
        plt.savefig(file_path, format='jpg', dpi=300, bbox_inches="tight")
    
        # Show the heatmap as a figure
        plt.show()
        plt.close()
    
    # Assuming combined_footprints is your DataFrame
    for name, footprints in F_dict_animal.items():
        combined_footprints = footprints['combined_footprints'].rename(columns={'Nongrazing': 'Croplands'})  # Rename column
        cleaned_name = name.replace(' ', '_')
    
        # Title for the heatmap, replacing underscores with spaces for display
        title = f'{cleaned_name.replace("_", " ")} - {animal_product.replace("_", " ")} {extension_name}'
    
        # Generate the filename and save the heatmap image
        filename = f'combined_footprints_{cleaned_name}.jpg'
        
        # Save the combined footprints heatmap as a JPG with the title and display it
        save_heatmap_as_image(combined_footprints, filename, title)
    
        print(f"Saved and displayed heatmap image for {cleaned_name}")
        
    # Company footprints heatmap
    
    # Generate and save heatmap for all_footprints_grouped
    save_heatmap_as_image(all_footprints_grouped, filename, title)
    
    print("Saved and displayed heatmap.")
    
    # Company footprints heatmap in kilo-unit
    # Convert all_footprints_grouped to div/1000
    all_footprints_grouped_div1k = all_footprints_grouped / 1000
    
    # Generate and save heatmap for all_footprints_grouped_div1k
    
    save_heatmap_as_image(all_footprints_grouped_div1k, filename_div1k, title_div1k)
    
    print("Saved and displayed heatmap for all_footprints_grouped_div1k.")
    
    # Export into excel file for final part
    excel_filename = f"{file_path_results}\\{file_name_animal}_footprints.xlsx"
    
    with pd.ExcelWriter(excel_filename, engine="xlsxwriter") as writer:
        all_footprints_grouped_div1k.to_excel(writer, sheet_name="Footprints", index=True)
        
        # Formatting for better readability
        workbook = writer.book
        worksheet = writer.sheets["Footprints"]
        worksheet.set_column("A:A", 20)  # Adjust column width for company names
        worksheet.set_column("B:Z", 15)  # Adjust column width for numbers
    
    print(f"Exported footprints data to {excel_filename}")
    
    # Company footprints heatmap
    # Define file paths
    excel_filename_unrounded = f"{file_path_results}{excel_export}_unrounded.xlsx"
    excel_filename_rounded = f"{file_path_results}{excel_export}_rounded.xlsx"
    
    # Create empty lists to store DataFrames
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
    
    # Save both versions to Excel
    for filename, df in [(excel_filename_unrounded, all_footprints_df_unrounded), (excel_filename_rounded, all_footprints_df_rounded)]:
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Regional_Footprints", index=False)
    
            # Formatting
            workbook = writer.book
            worksheet = writer.sheets["Regional_Footprints"]
            worksheet.set_column("A:A", 30)  # Company names
            worksheet.set_column("B:B", 15)  # Land type
            worksheet.set_column("C:Z", 15)  # Numerical values
    
        print(f"Exported footprints data to {filename}")
    
    # Plotting
    # --- Prepare data ---
    df_companies = all_footprints_grouped_div1k.copy()
    df_companies = df_companies.reindex(sorted(df_companies.index))
    df_companies.index = df_companies.index.str.replace("_", " ")
    
    df_companies = df_companies.rename(index={"Arab Company for Livestock Development (ACOLID)": "ACOLID"})
    
    # Totals (Mha)
    df_total = pd.DataFrame({
        "Grazing": [df_companies["Grazing"].sum() / 1000],
        "Croplands": [df_companies["Croplands"].sum() / 1000]
    }, index=["All companies"])
    
    # --- Colors ---
    color_grazing = "#5ab4ac" #"#87b6c2"
    color_cropland = "#d8b365"
    
    # --- Font sizes ---
    font_top = 10       # x-axis labels
    font_small = 8      # everything else
    
    # --- Bar heights ---
    bar_height_top = 0.6      # top bar
    bar_height_bottom = 0.8   # bottom bars
    
    # --- Figure setup ---
    fig_width = 9
    fig_height = 11
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(fig_width, fig_height),
        gridspec_kw={'height_ratios': [1, 8]}
    )
    
    # =====================
    # Top subplot: total footprint
    # =====================
    bottom = np.zeros(len(df_total))
    for col, color in zip(["Grazing", "Croplands"], [color_grazing, color_cropland]):
        ax_top.barh(df_total.index, df_total[col], left=bottom, color=color, alpha=0.9,
                    height=bar_height_top)
        bottom += df_total[col]
        
    # Reverse y-axis to match alphabetical order
    ax_bottom.invert_yaxis()
    
    # Top labels inside bars
    g_val = df_total["Grazing"].values[0]
    c_val = df_total["Croplands"].values[0]
    total_val = g_val + c_val
    x_pos = total_val + total_val * 0.01
    
    ax_top.text(x_pos, 0.04, f"{int(round(g_val)):,}", va="bottom", ha="left",
                fontsize=font_small, color=color_grazing, fontweight="bold")
    ax_top.text(x_pos, -0.04, f"{int(round(c_val)):,}", va="top", ha="left",
                fontsize=font_small, color=color_cropland, fontweight="bold")
    
    # X-axis title
    ax_top.set_xlabel(f"{graph_label_animal}{graph_label_fp}of all companies {graph_unit_top}",
                      fontsize=font_top, fontweight="bold")
    
    # Tick labels small
    ax_top.tick_params(axis='x', labelsize=font_small)
    ax_top.tick_params(axis='y', labelsize=font_small)
    
    ax_top.spines[['top', 'right']].set_visible(False)
    ax_top.grid(axis='x', linestyle=':', alpha=0.6)
    
    ax_top.set_axisbelow(True)
    ax_top.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    
    # =====================
    # Bottom subplot: individual companies
    # =====================
    bottom = np.zeros(len(df_companies))
    for col, color in zip(["Grazing", "Croplands"], [color_grazing, color_cropland]):
        ax_bottom.barh(df_companies.index, df_companies[col], left=bottom,
                       color=color, edgecolor="none", alpha=0.9, height=bar_height_bottom)
        bottom += df_companies[col]
        
        # Reverse y-axis to match alphabetical order
        ax_bottom.invert_yaxis()
    
    # Add numeric labels
    label_offset = 0.00000001         # has to be smaller rather than bigger since the yaxis is reversed to keep things alphabetical
    for y, (idx, row) in enumerate(df_companies.iterrows()):
        g, c = row["Grazing"], row["Croplands"]
        total = g + c
        x_pos = total + total * 0.01
    
        ax_bottom.text(x_pos, y + label_offset, f"{int(round(g)):,}", va="bottom", ha="left",
                       fontsize=font_small, color=color_grazing, fontweight="bold")
        ax_bottom.text(x_pos, y - label_offset, f"{int(round(c)):,}", va="top", ha="left",
                       fontsize=font_small, color=color_cropland, fontweight="bold")
    
    # Format x-axis
    ax_bottom.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    
    # Keep x-axis label size
    ax_bottom.set_xlabel(f"{graph_label_animal}{graph_label_fp}by company {graph_unit_bottom}",
                         fontsize=font_top, fontweight="bold")
    
    # Y-axis labels font
    ax_bottom.set_yticklabels(df_companies.index, fontsize=font_small)
    
    ax_bottom.spines[['top', 'right']].set_visible(False)
    ax_bottom.grid(axis='x', linestyle=':', alpha=0.6)
    ax_bottom.set_axisbelow(True)
    ax_bottom.tick_params(axis='x', labelsize=font_small)
    
    # =====================
    # Legend centered below
    # =====================
    handles = [
        plt.Line2D([0], [0], color=color_grazing, lw=6),
        plt.Line2D([0], [0], color=color_cropland, lw=6)
    ]
    labels = ["Grazing", "Croplands"]
    
    ax_bottom.legend(handles, labels, loc='upper center', ncol=2, frameon=False,
                     fontsize=font_small, bbox_to_anchor=(0.5, -0.08))
    
    plt.tight_layout(h_pad=4.0)  # space between the bar charts
    plt.show()

    # Prepare the company regional footprints dataframe for exporting
    # -------------------------------
    df_regions = all_footprints_df_rounded.copy()
    
    # Ensure there's an 'Animal' column
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
            raise ValueError("Couldn't find numeric footprint column.")
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
    
with pd.ExcelWriter(full_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    all_regions_df.to_excel(writer, sheet_name=animal_product, index=False)
    