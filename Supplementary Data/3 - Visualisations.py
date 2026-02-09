"""
Created on Wed Oct 22 11:47:58 2025 by Christina Drotenko

Purpose of code: creating figures
Input: results
Output: figures

"""

#%%
# Imports
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.ticker as mtick
import matplotlib as mpl
import seaborn as sns
import textwrap

# INSERT YOUR FILE PATH BELOW:
#file_path = r"INSERTHERE\Supplementary Data\data\3 - results"

totals_bar_chart = pd.read_excel(file_path + r"\Results_visualisations.xlsx", sheet_name="Summary for Figures")
totals_bar_chart = totals_bar_chart.drop(columns='Amount')  #drop the unrounded col
totals_bar_chart = totals_bar_chart.rename(columns={'Amount (rounded)':'Amount'})

poultry_meat = pd.read_excel(file_path + r"\Results_visualisations.xlsx", sheet_name="Poultry Meat")
bovine_meat = pd.read_excel(file_path + r"\Results_visualisations.xlsx", sheet_name="Bovine Meat")
pigmeat = pd.read_excel(file_path + r"\Results_visualisations.xlsx", sheet_name="Pigmeat")
dairy = pd.read_excel(file_path + r"\Results_visualisations.xlsx", sheet_name="Milk - Excluding Butter")

# for the animal types drop the unrounded cols (only keep the ones rounded to 3 significant digits)
poultry_meat = poultry_meat.drop(columns='Footprint amount')  #drop the unrounded col
poultry_meat = poultry_meat.rename(columns={'Footprint amount (rounded)':'Footprint amount'})

bovine_meat = bovine_meat.drop(columns='Footprint amount')  #drop the unrounded col
bovine_meat = bovine_meat.rename(columns={'Footprint amount (rounded)':'Footprint amount'})

pigmeat = pigmeat.drop(columns='Footprint amount')  #drop the unrounded col
pigmeat = pigmeat.rename(columns={'Footprint amount (rounded)':'Footprint amount'})

dairy = dairy.drop(columns='Footprint amount')  #drop the unrounded col
dairy = dairy.rename(columns={'Footprint amount (rounded)':'Footprint amount'})

# Make df for all impacts
all_impacts = pd.concat([poultry_meat, bovine_meat, pigmeat, dairy], ignore_index=True)

# Replace full ACOLID name with shortened version
all_impacts.loc[all_impacts['Company'] == 'Arab Company for Livestock Development (ACOLID)', 'Company'] = 'ACOLID'


# First adjust the units to become smaller and then change the unit title
#1,000m^3 = div by another 1,000,000
all_impacts.loc[all_impacts['Unit'] == '(1,000 m³)', 'Footprint amount'] /= 1000000
all_impacts.loc[all_impacts['Unit'] == '(1,000 m³)', 'Unit'] = '(km³)'

#make kha into Mha --> div by another thousand
all_impacts.loc[all_impacts['Unit'] == '(kha)', 'Footprint amount'] /= 1000
all_impacts.loc[all_impacts['Unit'] == '(kha)', 'Unit'] = '(Mha)'

#make t into kt = div by another 1,000
all_impacts.loc[all_impacts['Unit'] == '(t)', 'Footprint amount'] /= 1000
all_impacts.loc[all_impacts['Unit'] == '(t)', 'Unit'] = '(kt)'


# Create a version of the excel with combined continents for the visualisations (non-GLEAM regions)
region_mapping = {
    "East Asia and Southeast Asia": "Asia",
    "South Asia": "Asia",
    "Eastern Europe": "Europe",
    "Western Europe": "Europe",
    "Russian Federation": "Asia",
    "Latin America and the Caribbean": "South America",
    "Near East and North Africa": "Middle East and Africa",
    "North America": "North America",
    "Oceania": "Oceania",
    "Sub-Saharan Africa": "Middle East and Africa",
    "RoW": "RoW"
}

all_impacts_agg = all_impacts.assign(Region=all_impacts["Region"].map(region_mapping))

# Groupby function by region (based on the region mapping above)
all_impacts_agg = all_impacts_agg.groupby(["Company", "Land type", "Animal product", "Region", "Footprint", "Unit"],as_index = False)["Footprint amount"].sum()

# Round values in the above to be to 3 significant digits

# Combine blue and green water and then add these new rows into a main df
water_df = all_impacts_agg.loc[all_impacts_agg['Footprint'].isin(['Blue water', 'Green water'])
                           ].groupby(['Company', 'Land type','Animal product','Region','Unit'], as_index=False)['Footprint amount'].sum()

# Rename the footprint column and re-order columns
water_df['Footprint'] = 'Blue and green water'

# Keep other impacts
otherfp_df = all_impacts_agg.loc[~all_impacts_agg['Footprint'].isin(['Blue water', 'Green water'])]

# Combine
all_impacts_df = pd.concat([otherfp_df, water_df], ignore_index=True)
all_impacts_df = all_impacts_df[['Company','Land type','Animal product','Region','Footprint','Unit','Footprint amount']]


#%% CREATE PIVOT FOR HEATMAP

# Make a df in the format of the heatmap (based on all_impacts_agg) with:
    # Rows as the companies
    # Columns as the footprint / animal categories
    
heatmap_data = all_impacts_df.copy()

# Group by company
heatmap_data = heatmap_data.groupby(["Company", "Animal product", "Footprint", "Unit"],as_index = False)["Footprint amount"].sum()

# Create pivot to match the heatmap format
heatmap_pivot = heatmap_data.pivot(
    index="Company",
    columns=["Animal product", "Footprint"],
    values="Footprint amount")

# Re-order for plotting to have footprints together instead of animal products

fp_order=["Land use,","Blue and green water","Nitrogen application"]

heatmap_pivot = heatmap_pivot.sort_index(
    axis=1,
    level=1,
    key=lambda x: pd.Categorical(x.tolist(), categories=fp_order, ordered=True)
)

# Normalise each column (0–1 per column) so the heatmap is compared per column
heatmap_norm = heatmap_pivot.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

#%% Heatmaps transposed with separate heatmaps for each - FINAL FIGURE 1B
# Split up heatmap_norm into 4 dfs based on animal category + drop rows with nans
heatmap_bovine_arranged = heatmap_norm.xs("Bovine Meat", level=0, axis=1).dropna(how="any")
heatmap_poultry_arranged = heatmap_norm.xs("Poultry Meat", level=0, axis=1).dropna(how="any")
heatmap_pigmeat_arranged = heatmap_norm.xs("Pigmeat", level=0, axis=1).dropna(how="any")
heatmap_dairy_arranged = heatmap_norm.xs("Milk - Excluding Butter", level=0, axis=1).dropna(how="any")

# Reorder the rows of the heatmap
new_order = ["Land use", "Blue and green water", "Nitrogen application"]
heatmap_bovine = heatmap_bovine_arranged[new_order]
heatmap_poultry = heatmap_poultry_arranged[new_order]
heatmap_pigmeat = heatmap_pigmeat_arranged[new_order]
heatmap_dairy = heatmap_dairy_arranged[new_order]

heatmaps = [
    ("A) Beef", heatmap_bovine),
    ("B) Poultry", heatmap_poultry),
    ("C) Pork", heatmap_pigmeat),
    ("D) Dairy", heatmap_dairy)
]

cbar_label = "Footprint scale"
cmap = "Greys"

cell_width = 0.6   # width of each col
cell_height = 1  # height of each row


# Make subplot
fig, axes = plt.subplots(2,2,figsize=(20, 18))

axes = axes.flatten()


# Let it loop
for ax, (title, df) in zip(axes, heatmaps):

    sns.heatmap(
        df,
        ax=ax,
        cmap=cmap,
        cbar=False,
        annot=False,
        square=False
    )
    
    # Draw black border around heatmap cells only
    n_rows, n_cols = df.shape
    
    rect = Rectangle(
        (0, 0),                # lower-left corner in data coords
        n_cols,                # width
        n_rows,                # height
        fill=False,
        edgecolor="black",
        linewidth=1.5
    )
    
    ax.add_patch(rect)

    ax.set_title(title, fontsize=16, loc='left', pad=10, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    
    # Wrap x-axis labels
    wrapped_x = [
        textwrap.fill(label.get_text(), width=14)
        for label in ax.get_xticklabels()
    ]
    ax.set_xticklabels(wrapped_x, rotation=0, ha='center', fontsize=14)
    ax.tick_params(axis='x', which='both', length=0)  # hide the tick marks

    # Wrap y-axis labels
    wrapped_y = [
        textwrap.fill(label.get_text(), width=25)
        for label in ax.get_yticklabels()
    ]
    ax.set_yticklabels(wrapped_y, rotation=0, ha='right', fontsize=14)

    ax.tick_params(axis='y', labelsize=18)
    
# Adjust colourbar
cbar_ax = fig.add_axes([0.155, -0.02, 0.835, 0.02])    # add axis below subplots - left, bottom, width, height for adjustments
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm,cax=cbar_ax,orientation='horizontal')

cbar.ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: f"{x*100:.0f}%"))
cbar.set_label("Normalised impact score (%)", fontsize=18)
cbar.ax.tick_params(labelsize=18)
plt.tight_layout()
plt.show()



#%% ABSOLUTE FOOTPRINT FOR ANIMAL CATEGORY CHART - FINAL FIGURE 2A

# FIRST PROPORTIONALLY (ADD UP TO 100)
# In the original df calculate the totals by the animal product across all the footprints

# Create 3 fp dfs with these numbers (with all animals in each)
animals_graph_land = otherfp_df.loc[otherfp_df['Footprint'].isin(['Land use'])]
animals_graph_water = water_df.copy()
animals_graph_nitrogen = otherfp_df.loc[otherfp_df['Footprint'].isin(['Nitrogen application'])]

# Replace terminology for "Milk - Excluding Butter" with "Dairy", "Pigmeat" to be "Pork", "Bovine Meat" to be "Beef", and "Poultry Meat" to be "Poultry"
animals_graph_land.loc[:, "Animal product"] = animals_graph_land["Animal product"].replace({
    "Milk - Excluding Butter": "Dairy",
    "Pigmeat": "Pork",
    "Bovine Meat": "Beef",
    "Poultry Meat": "Poultry"})

animals_graph_water.loc[:, "Animal product"] = animals_graph_water["Animal product"].replace({
    "Milk - Excluding Butter": "Dairy",
    "Pigmeat": "Pork",
    "Bovine Meat": "Beef",
    "Poultry Meat": "Poultry"})

animals_graph_nitrogen.loc[:, "Animal product"] = animals_graph_nitrogen["Animal product"].replace({
    "Milk - Excluding Butter": "Dairy",
    "Pigmeat": "Pork",
    "Bovine Meat": "Beef",
    "Poultry Meat": "Poultry"})


# Calculate the sums of each to be per animal
animals_graph_land_sum = animals_graph_land.groupby(['Animal product', 'Footprint','Unit'])['Footprint amount'].sum().reset_index()
animals_graph_water_sum = animals_graph_water.groupby(['Animal product', 'Footprint','Unit'])['Footprint amount'].sum().reset_index()
animals_graph_nitrogen_sum = animals_graph_nitrogen.groupby(['Animal product', 'Footprint','Unit'])['Footprint amount'].sum().reset_index()

# Combine these dfs into one via concatenation
animals_graph_all = pd.concat([animals_graph_land_sum, animals_graph_water_sum, animals_graph_nitrogen_sum], ignore_index=True)

# Put the summed animals in chronological order from largest to smallest per impact - this sorts first by footprint, then by footprint amount
animals_graph_all = animals_graph_all.sort_values(['Footprint','Footprint amount'], ascending=[True, False])

# Round the values here to be to three significant digits for the graphs
animals_graph_all["Footprint amount"] = (animals_graph_all["Footprint amount"].apply(lambda x: float(f"{x:.3g}")))

### Make a bar chart with three different colours per line and with different axes
# Pivot dataframe
df_pivot = animals_graph_all.pivot(index='Footprint', columns='Animal product', values='Footprint amount')

# Footprints in desired order
footprint_order = ['Land use', 'Blue and green water', 'Nitrogen application']
df_pivot = df_pivot.loc[footprint_order]

# Colours for each subplot https://colorbrewer2.org/#type=diverging&scheme=PuOr&n=4
footprint_colors = ['#a6dba0', '#92c5de', '#b2abd2'] # green, blue, purple - colourblind safe
# other options:
#['#66c2a5', '#3288bd', '#fdae61'] # green, blue, orange
#['#66c2a5', '#3288bd', '#5e4fa2']  # green, blue, purple

# Create manual label for shortened region names
region_labels = {
    "North America": "NAM",
    "South America": "SAM",
    "Europe": "EUR",
    "Asia": "ASI",
    "Oceania": "OCE",
    "Middle East and Africa": "MEA",
    "RoW": "RoW"
}

regions = df_pivot.columns.tolist()

# Create a function
def plot_animals_bar(df_pivot, animals_graph_all, footprint_order, footprint_colors, region_labels, show_labels=True):
    fig, axes = plt.subplots(len(footprint_order), 1, figsize=(12, 3.5))
    plt.subplots_adjust(hspace=0.4)
    
    # 1 - First plot with written labels (no icons)
    for i, footprint in enumerate(footprint_order):
        ax = axes[i]
        unit = animals_graph_all.loc[
            animals_graph_all['Footprint'] == footprint, 'Unit'].iloc[0]
        
        bottom = 0
        color = footprint_colors[i]
        
        # Sort regions by footprint size (left biggest, right smallest)
        region_order = df_pivot.loc[footprint].sort_values(ascending=False).index
        
        # Remove "RoW" just for this chart (it is too small)
        region_order = [r for r in region_order if r != "RoW"]
            
        for region in region_order:
            width = df_pivot.loc[footprint, region]
            # Draw horizontal bar
    
            ax.barh(
                y=0,
                width=width,
                left=bottom,
                height=0.6,
                color=color,
                edgecolor='white',
                linewidth=2
            )
    
            label_y = 0.1  # above the bar
            va = 'bottom'
            num_y = label_y - 0.03  # number below the label
            num_va = 'top'
            perc_y = num_y - 0.14   # percentage below numeric value
           
            # Use shortened label for the text
            manual_region_label = region_labels.get(region, region)  # fallback to original if not in dictionary

            # Region label
            ax.text(
                x=bottom + width / 2,
                y=label_y,
                s=manual_region_label,
                va=va,
                ha='center',
                color='black',
                fontweight='bold',
                fontsize=12
            )
            
            # Place numerical amount below the label
            ax.text(
                x=bottom + width / 2,
                y=num_y,
                s=f"{width:,.0f}",
                va=num_va,
                ha='center',
                color='black',
                fontsize=10
            )
            
            # Percentage of total for fp
            total = df_pivot.loc[footprint].sum()
            perc = width / total * 100
            ax.text(
                x=bottom + width / 2,
                y=perc_y,
                s=f"{perc:.0f}%",       # 1 decimal place for the %s because blue/green water totals 101% otherwise due to rounding
                va='top',
                ha='center',
                color='black',
                fontsize=10
            )
    
            
            bottom += width
    
        ax.set_xlim(0, bottom)
        ax.set_ylim(-0.2,0.3)
        
        # Add vertical grid lines
        ax.grid(axis='x', which='major', color='grey', linestyle='-', linewidth=0.4, alpha=0.7)
        ax.grid(axis='x', which='minor', color='grey', linestyle='--', linewidth=0.3, alpha=0.4)
        ax.minorticks_on()
        
        # Add the vertical zero line
        ax.axvline(0, color='black', linewidth=1)
    
        ax.set_yticks([])
        ax.set_ylabel(textwrap.fill(f'{footprint} footprint {unit}', width=12),
                      fontsize=10,
                      rotation=0, #horiz text
                      va='center',
                      ha='right')
            #f'{footprint} footprint {unit}', fontsize=8)
        
        
        # Add commasto x-axis ticks
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    
    
    
    sns.despine(left=True)
    plt.tight_layout()
    plt.show()

plot_animals_bar(df_pivot, animals_graph_all, footprint_order, footprint_colors, region_labels, show_labels=True)

# 2 - Then plot without the names of the products (for the icons in draw.io) - first make a function and then just run it (kept the function because need it for other work)

def plot_animals_bar_no_label(df_pivot, animals_graph_all, footprint_order, footprint_colors, region_labels, show_labels=True):
    fig, axes = plt.subplots(len(footprint_order), 1, figsize=(12, 3.5))
    plt.subplots_adjust(hspace=0.4)
    
    
    for i, footprint in enumerate(footprint_order):
        ax = axes[i]
        unit = animals_graph_all.loc[
            animals_graph_all['Footprint'] == footprint, 'Unit'].iloc[0]
        
        bottom = 0
        color = footprint_colors[i]
        
        # Sort regions by footprint size (left biggest, right smallest)
        region_order = df_pivot.loc[footprint].sort_values(ascending=False).index
        
        # Remove "RoW" just for this chart (it is too small)
        region_order = [r for r in region_order if r != "RoW"]
            
        for region in region_order:
            width = df_pivot.loc[footprint, region]
            # Draw horizontal bar
    
            ax.barh(
                y=0,
                width=width,
                left=bottom,
                height=0.6,
                color=color,
                edgecolor='white',
                linewidth=2
            )
    
            label_y = 0.1  # above the bar
            num_y = label_y - 0.03  # number below the label
            num_va = 'top'
            perc_y = num_y - 0.14   # percentage below numeric value
               

            # Place numerical amount below the label
            ax.text(
                x=bottom + width / 2,
                y=num_y,
                s=f"{width:,.0f}",
                va=num_va,
                ha='center',
                color='black',
                fontsize=10
            )
            
            # Percentage of total for fp
            total = df_pivot.loc[footprint].sum()
            perc = width / total * 100
            ax.text(
                x=bottom + width / 2,
                y=perc_y,
                s=f"{perc:.0f}%",
                va='top',
                ha='center',
                color='black',
                fontsize=10
            )
    
            
            bottom += width
    
        ax.set_xlim(0, bottom)
        ax.set_ylim(-0.2,0.3)
        
        # Add vertical grid lines
        ax.grid(axis='x', which='major', color='grey', linestyle='-', linewidth=0.4, alpha=0.7)
        ax.grid(axis='x', which='minor', color='grey', linestyle='--', linewidth=0.3, alpha=0.4)
        ax.minorticks_on()
        
        # Add the vertical zero line
        ax.axvline(0, color='black', linewidth=1)
        
        ax.set_yticks([])
        ax.set_ylabel(textwrap.fill(f'{footprint} footprint {unit}', width=12),
                      fontsize=10,
                      rotation=0, #horiz text
                      va='center',
                      ha='right')
            #f'{footprint} footprint {unit}', fontsize=8)
        
        
        # Format xaxis ticks with commas
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
        ax.tick_params(axis='x', labelsize=10)
    
        
    
    sns.despine(left=True)
    plt.tight_layout()
    plt.show()

plot_animals_bar_no_label(df_pivot, animals_graph_all, footprint_order, footprint_colors, region_labels, show_labels=True)


#%% PROPORTIONAL FOOTPRINT FOR REGIONS CHART - FINAL FIGURE 2B

# Calculate the sums of each to be per region based on the dfs in the section above
regions_graph_land_sum = animals_graph_land.groupby(['Region', 'Footprint','Unit'])['Footprint amount'].sum().reset_index()
regions_graph_water_sum = animals_graph_water.groupby(['Region', 'Footprint','Unit'])['Footprint amount'].sum().reset_index()
regions_graph_nitrogen_sum = animals_graph_nitrogen.groupby(['Region', 'Footprint','Unit'])['Footprint amount'].sum().reset_index()

# Combine these dfs into one via concatenation
regions_graph_all = pd.concat([regions_graph_land_sum, regions_graph_water_sum, regions_graph_nitrogen_sum], ignore_index=True)

# Put the summed animals in chronological order from largest to smallest per impact - this sorts first by footprint, then by footprint amount
regions_graph_all = regions_graph_all.sort_values(['Footprint','Footprint amount'], ascending=[True, False])

### Make a graph
# Pivot dataframe
df_pivot = regions_graph_all.pivot(index='Footprint', columns='Region', values='Footprint amount')

# Footprints in desired order
footprint_order = ['Land use', 'Blue and green water', 'Nitrogen application']
df_pivot = df_pivot.loc[footprint_order]

# Colors for each subplot https://colorbrewer2.org/#type=diverging&scheme=PuOr&n=4
footprint_colors = ['#a6dba0', '#92c5de', '#b2abd2'] # green, blue, purple - colourblind safe
# other options:
#['#66c2a5', '#3288bd', '#fdae61'] # green, blue, orange
#['#66c2a5', '#3288bd', '#5e4fa2']  # green, blue, purple

# Create manual label for shortened region names w/3 letters
region_labels = {
    "North America": "NAM",
    "South America": "SAM",
    "Europe": "EUR",
    "Asia": "ASI",
    "Oceania": "OCE",
    "Middle East and Africa": "MEA",
    "RoW": "RoW"
}

regions = df_pivot.columns.tolist()

fig, axes = plt.subplots(len(footprint_order), 1, figsize=(12, 3.5))
plt.subplots_adjust(hspace=0.4)

# 1 - First plot with written labels (no icons)
for i, footprint in enumerate(footprint_order):
    ax = axes[i]
    unit = animals_graph_all.loc[
        animals_graph_all['Footprint'] == footprint, 'Unit'].iloc[0]
    
    bottom = 0
    color = footprint_colors[i]
    
    # Sort regions by footprint size (left biggest, right smallest)
    region_order = df_pivot.loc[footprint].sort_values(ascending=False).index
    
    # Remove "RoW" just for this chart (it is too small)
    region_order = [r for r in region_order if r != "RoW"]

    label_y = 0.1  # above the bar
    va = 'bottom'
    num_y = label_y - 0.03  # number below the label
    num_va = 'top'
    perc_y = num_y - 0.14   # percentage below numeric value
       
        
    
    for idx, region in enumerate(region_order):
        width = df_pivot.loc[footprint, region]
        is_last=(idx==len(region_order)-1)  #making the one on the right be off a bit to the side
        
        # set manual shortened labels
        manual_region_label = region_labels.get(region,region)
        
        # Draw horizontal bar

        ax.barh(
            y=0,
            width=width,
            left=bottom,
            height=0.6,
            color=color,
            edgecolor='white',
            linewidth=2
        )

        # create special case for the rightmost region to be a bit off to the side
        if is_last:
            text_x=bottom+width+((df_pivot.loc[footprint].sum())*0.01)
            text_ha='left'
        else:
            text_x=bottom+width/2
            text_ha='center'
            
        # now apply specific labels to specific coordinates
        # Region label
        ax.text(
            x=text_x,
            y=label_y,
            s=manual_region_label,
            va=va,
            ha=text_ha,
            color='black',
            fontweight='bold',
            fontsize=12
        ) 

        # Place numerical amount below the label
        ax.text(
            x=text_x,
            y=num_y,
            s=f"{width:,.0f}",
            va=num_va,
            ha=text_ha,
            color='black',
            fontsize=10
        )
        
        # % of total for fp
        total = df_pivot.loc[footprint].sum()
        perc = width / total * 100
        ax.text(
            x=text_x,
            y=perc_y,
            s=f"{perc:.0f}%",       # 1 decimal place for the %s because blue/green water totals 101% otherwise due to rounding
            va='top',
            ha=text_ha,
            color='black',
            fontsize=10
        )

        
        bottom += width

    ax.set_xlim(0, bottom)
    ax.set_ylim(-0.2,0.3)
    
    # Add vertical grid lines
    ax.grid(axis='x', which='major', color='grey', linestyle='-', linewidth=0.4, alpha=0.7)
    ax.grid(axis='x', which='minor', color='grey', linestyle='--', linewidth=0.3, alpha=0.4)
    ax.minorticks_on()
    
    # Add the vertical zero line
    ax.axvline(0, color='black', linewidth=1)

    ax.set_yticks([])
    ax.set_ylabel(textwrap.fill(f'{footprint} footprint {unit}', width=12),
                  fontsize=10,
                  rotation=0, #horiz text
                  va='center',
                  ha='right')
        #f'{footprint} footprint {unit}', fontsize=8)
    
    
    # Format x-axis ticks with commas
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    ax.tick_params(axis='x', labelsize=10)



sns.despine(left=True)
plt.tight_layout()
plt.show()

#%% Calculations for paper numbers - to be included in SI

# Relative amounts per company (that they contribute to the total for their impact)
heatmap_relative_amt_land = heatmap_data.copy()
heatmap_relative_amt_land = heatmap_relative_amt_land.loc[
    heatmap_relative_amt_land["Footprint"] == "Land use"]
heatmap_relative_amt_land["Relative to all fps"] = (                    #relative % to all
    heatmap_relative_amt_land["Footprint amount"]
    / heatmap_relative_amt_land["Footprint amount"].sum()* 100)
heatmap_relative_amt_land["Relative to animal product"] = (             #relative % to animal prod only
    heatmap_relative_amt_land
    .groupby("Animal product")["Footprint amount"]
    .transform(lambda x: x / x.sum() * 100)
)

heatmap_relative_amt_water = heatmap_data.copy()
heatmap_relative_amt_water = heatmap_relative_amt_water.loc[
    heatmap_relative_amt_water["Footprint"] == "Blue and green water"]
heatmap_relative_amt_water["Relative to all fps"] = (                    #relative %
    heatmap_relative_amt_water["Footprint amount"]
    / heatmap_relative_amt_water["Footprint amount"].sum()* 100)
heatmap_relative_amt_water["Relative to animal product"] = (             #relative % to animal prod only
    heatmap_relative_amt_water
    .groupby("Animal product")["Footprint amount"]
    .transform(lambda x: x / x.sum() * 100)
)

heatmap_relative_amt_nitrogen = heatmap_data.copy()
heatmap_relative_amt_nitrogen = heatmap_relative_amt_nitrogen.loc[
    heatmap_relative_amt_nitrogen["Footprint"] == "Nitrogen application"]
heatmap_relative_amt_nitrogen["Relative to all fps"] = (                    #relative %
    heatmap_relative_amt_nitrogen["Footprint amount"]
    / heatmap_relative_amt_nitrogen["Footprint amount"].sum()* 100)
heatmap_relative_amt_nitrogen["Relative to animal product"] = (             #relative % to animal prod only
    heatmap_relative_amt_nitrogen
    .groupby("Animal product")["Footprint amount"]
    .transform(lambda x: x / x.sum() * 100)
)