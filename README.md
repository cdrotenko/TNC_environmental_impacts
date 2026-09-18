# TNC environmental impacts
# Modelling steps and data structure

## Step 1: Download the raw data
Download the necessary raw data from the following sources:

| Description                                      | Source                          | Availability                                                                                     |
|--------------------------------------------------|---------------------------------|-------------------------------------------------------------------------------------------------|
| FABIO database (2020)                             | Bruckner et al., 2019           | Can be downloaded here [https://zenodo.org/records/2577067](https://zenodo.org/records/2577067) |
| Nitrogen application                              | Bruckner et al., 2019           | Can be downloaded here [https://github.com/fineprint-global/fabio](https://github.com/fineprint-global/fabio) |
| Blue water use                                    | Bruckner et al., 2019           | Can be downloaded here [https://github.com/fineprint-global/fabio](https://github.com/fineprint-global/fabio) |
| Green water use (2020)                            | Bruckner et al., 2019           | Can be downloaded here [https://github.com/fineprint-global/fabio](https://github.com/fineprint-global/fabio) |
| Land use (2020)                                   | Liu et al., 2026                | Downloadable from the publication here [https://www.nature.com/articles/s43016-026-01387-0] |
| Global livestock processing in heads/tonnes (2022, retrieved June 23, 2025)| Food and Agriculture Organization| Downloadable from our repository or can be found here [https://www.fao.org/faostat/en/#data](https://www.fao.org/faostat/en/#data). File name: Global processing.xlsx |
| TNC processing (2021-2023)                             | Profundo, followed by additional processing from authors | Downloadable from our GitHub repository in the ‘1 – raw data’ folder. File name: TNC processing.xlsx |
| Concordance matrix                                | Authors                         | Used to align FABIO countries with GLEAM regions. Downloadable from our GitHub repository in the ‘1 – raw data’ folder. File name: Concordance.xlsx |

## Step 2: Activate environment
See the included ‘environment.yml’ file for the module requirements.

## Step 3: Process the data and calculate the results
The total expected package installation time is ~2 minutes and runtime is ~15 minutes. Run the ‘1 – Processing data.py’ file to first process the data. Then, run ‘2 – Model.py’ to calculate the necessary results. Beyond the processing completed in the code, some files are processed via Excel for which the files are already in the corresponding folders ‘2 – processed data’ and ‘3 – results’. The other files are created from the ‘1 – Processing data.py’ file. See these file descriptions for both below:

| Description                                      | Availability and file name                                                                                     |
|--------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
|FABIO database (formatted data)	|Making the FABIO data ready for our model. Downloadable from our repository in the ‘2 – processed data’ folder. File names: x_ungrouped.csv, Y_ungrouped.csv, Z_ungrouped.csv, F_ungrouped.csv, blue_water.csv, green_water.csv, N_application.csv|
|TNC processing by weight	|TNC processing amounts converted into weight for beef, poultry, and pork. Downloadable from our repository in the ‘2 – processed data’ folder. File name: TNC processing by weight.xlsx |
|TNC environmental impacts	|The results from our model covering all the environmental impacts across the livestock products (by company, land type, region, impact amount, animal product, impact type, and unit across the four animal products). Downloadable from our repository in the ‘3 – results’ folder. File name: Results_all_impacts.xlsx (Please note that some additional processing in additional tabs is conducted in Excel before being used for the figures, which is under the file name Results_visualisations.xlsx under tabs ‘All TNC Impacts’, ‘Bovine Meat’, ‘Poultry Meat’, ‘Pigmeat’)|
|Relative global share of TNC processing|	Calculations of the TNCs’ processing relative to the global processing of these livestock products. Downloadable from our repository in the ‘3 – results’ folder. File name: Results_visualisations.xlsx under tab ‘Relative Processing’.|
|Global environmental impacts|	Global environmental impacts per livestock product calculated with FABIO, with the code in part of the 2 – Model.py file (for total global processing of each product and impact type). Downloadable from our repository in the ‘3 – results’ folder. File name: Results_visualisations.xlsx under tab ‘Global Impacts’|
|Relative global share of TNC impacts|	Calculations of the TNCs’ environmental impacts relative to the global environmental impacts for these livestock products, with a combined summary file of both the relative processing and relative impacts of the TNCs included under a separate tab. Downloadable from our repository in the ‘3 – results’ folder. File name: Results_visualisations.xlsx under tabs ‘Relative Impacts’ and ‘Summary for Figures’ (Note: the code for this is in the 3 – Visualisations.py file)|

## Step 4: Create figures
Run ‘3 – Visualisations.py’. Some minor formatting adjustments such as adding icons are completed using draw.io.

<div align="center">
  <img width="650" height="329" alt="image" src="https://github.com/user-attachments/assets/e0d59b0c-9e12-4e9a-bb93-7834e1938c93" />
</div>
