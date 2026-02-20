# TNC environmental impacts
# Modelling steps and data structure
<div align="center">
  <img width="650" height="329" alt="image" src="https://github.com/user-attachments/assets/eabcba56-ae4f-4c3e-b0cc-31701fa5f8f9" />
</div>

## Step 1: Download the raw data
Download the necessary raw data from the following sources:

| Description                                      | Source                          | Availability                                                                                     |
|--------------------------------------------------|---------------------------------|-------------------------------------------------------------------------------------------------|
| FABIO database (2020)                             | Bruckner et al., 2019           | Can be downloaded here [https://zenodo.org/records/2577067](https://zenodo.org/records/2577067) |
| Nitrogen application                              | Bruckner et al., 2019           | Can be downloaded here [https://github.com/fineprint-global/fabio](https://github.com/fineprint-global/fabio) |
| Blue water use                                    | Bruckner et al., 2019           | Can be downloaded here [https://github.com/fineprint-global/fabio](https://github.com/fineprint-global/fabio) |
| Green water use (2020)                            | Bruckner et al., 2019           | Can be downloaded here [https://github.com/fineprint-global/fabio](https://github.com/fineprint-global/fabio) |
| Land use (2020)                                   | Baoxiao et al., 2025            | Pre-print can be found here [https://www.researchsquare.com/article/rs-5527595/v1](https://www.researchsquare.com/article/rs-5527595/v1) and is still under review, but data can be used once it is published |
| Global livestock production in heads/tonnes (2022)| Food and Agriculture Organization| Downloadable from our repository or can be found here [https://www.fao.org/faostat/en/#data](https://www.fao.org/faostat/en/#data). File name: Global production.xlsx |
| TNC production (2022)                             | Profundo, followed by additional processing from authors | Downloadable from our GitHub repository in the ‘1 – raw data’ folder. File name: TNC production.xlsx |
| Concordance matrix                                | Authors                         | Used to align FABIO countries with GLEAM regions. Downloadable from our GitHub repository in the ‘1 – raw data’ folder. File name: Concordance.xlsx |

## Step 2: Activate environment
See the included ‘environment.yml’ file for the module requirements.

## Step 3: Process the data and calculate the results
Run the ‘1 – Processing data.py’ file to first process the data. Then, run ‘2 – Model.py’ to calculate the necessary results. Beyond the processing completed in the code, some files are processed via Excel for which the files are already in the corresponding folders ‘2 – processed data’ and ‘3 – results’. The other files are created from the ‘1 – Processing data.py’ file. See these file descriptions for both below:

| Description                                      | Availability and file name                                                                                     |
|--------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| TNC production by weight                         | TNC production converted into weight for beef, poultry, and pork. Downloadable from our repository in the ‘2 – processed data’ folder. File name: TNC production by weight.xlsx |
| Processed data from FABIO                        | Making the FABIO data ready for our model. Downloadable from our repository in the ‘2 – processed data’ folder. File names: x_ungrouped.csv, Y_ungrouped.csv, Z_ungrouped.csv, F_ungrouped.csv, blue_water.csv, green_water.csv, N_application.csv |
| TNC environmental impacts                        | The results from our model covering all the environmental impacts across the livestock products. Downloadable from our repository in the ‘3 – results’ folder. File name: Results_all_impacts.xlsx. Please note that some additional processing in additional tabs is conducted in Excel before being used for the figures, which is under the file name Results_visualisations.xlsx under tabs ‘All TNC Impacts’, ‘Bovine Meat’, ‘Poultry Meat’, ‘Pigmeat’ |
| Global environmental impacts                     | Global environmental impacts per livestock product calculated with FABIO, with the code in part of the 2 – Model.py file. Downloadable from our repository in the ‘3 – results’ folder. File name: Results_visualisations.xlsx under tab ‘Global Impacts’ |
| Relative global share of TNC production          | Calculations of the TNCs’ production relative to the global production of these livestock products. Downloadable from our repository in the ‘3 – results’ folder. File name: Results_visualisations.xlsx under tab ‘Relative Production’ |
| Relative global share of TNC impacts             | Calculations of the TNCs’ environmental impacts relative to the global environmental impacts for these livestock products. Downloadable from our repository in the ‘3 – results’ folder. File name: Results_visualisations.xlsx under tab ‘Relative Impacts’. Note: the code for this is in the 3 – Visualisations.py file. |
| Summary for Figures                              | Summary of the relative production and relative impacts of the TNCs to the world for use in the figures. |

## Step 4: Create figures
Run ‘3 – Visualisations.py’. Some minor formatting adjustments such as adding icons are completed using draw.io.

