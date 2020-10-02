#!/usr/bin/env python
# coding: utf-8

# <a id='top'></a>
# 
# ## 0. Introduction
# For a detailed introduction, check out my starter kernel [here](https://www.kaggle.com/dsholes/confused-start-here/). To skip straight to the juicy, plotly plots, click [here](#juicy). To get right into the code, click [here](#code)
# 
# I decided to focus my efforts on a methodology for **overlaying the census tract data with the police districts Use of Force (UOF) data**. As in my starter kernel, I'm only providing analysis for Dept_37-00027 (Austin, TX), but the methodology is applicable for all of the departments.
# 
# ## 1. Downloading Census Data
# 
# One thing to note is that the census data provided in the `cpe-data` folder sometimes doesn't cover all of the police districts. For example, the census data is just for one county (Travis County), but Dept_37-00027 spans multiple counties. Therefore, I'll "briefly" cover how to properly search for and download data from the Census website.
# 
# - Go to the [ACS website](
# https://www.census.gov/acs/www/data/data-tables-and-tools/data-profiles/2016/)
# - Select the U.S. State, e.g. Texas, that the police department is in.
# - Click the "Get Data Profile" button
# - Click "Demographic Characteristics"
# - Click "Add/Remove Geography"
# - "Select a geographic type" > "Census Tract - 140"
# - "Select a state" > "Texas"
# - Click "All Census Tracts within Texas"
# - Click "Add to your Selections"
# - Click "Show Table"
# - Click "Modify Table"
# - Click "Transpose Rows/Columns"
# - A side menu on the left allows you to select a year (I'm using 2014 to align with the Use of Force data)
# - Click "Download"
# - Select "Use Data"
# - Uncheck "Merge the annotations and data into a single file"
# - Uncheck "Include descriptive data element names"
# - Click "OK"
# - Click "Download"
# - Rejoice that you don't have to use this website every day
# 
# One nice thing about the American FactFinder website is that you can save a "Query" file, so you and your friends can avoid the 18 steps above if you ever need to re-download the data. To save the "Query" file, once the Table is formatted just like you want it:
# 
# - Click "Bookmark/Save" > "Save Query"
# 
# A `.aff` file will be downloaded. To use this file in the future, go to the [Advanced Search page](https://factfinder.census.gov/faces/nav/jsf/pages/searchresults.xhtml?refresh=t) of the American FactFinder website, and click "load search". Choose the `.aff` file and hit "OK".
# 
# ## 2. Methodology
# 
# Since the UOF data for Dept_37-00027 has `SUBJECT_RACE`, AND since we've just downloaded a fresh "Demographic Characteristics" file from the Census, let's use race as our example for how to overlay census data, that's traditionally served up in tract form, onto police districts, or `SECTOR`s as it's called in the Dept_37-00027 UOF data file.
# 
# The basic idea is to add up the populations (by race) of all of the census tracts contained within a police district (or SECTOR). This gets complicated by the fact that some census tracts are not contained 100% within a police district. Therefore, we have to somehow break up the populations of certain census tracts and re-distribute the populations. The obvious way, given the shapefiles, is to use areal fractions of the census tracts within the police districts. There are flaws associated with this. It (incorrectly) assumes:
# - that population is distributed equally among the entire area of the tract, and 
# - the racial populations have the same distribution across any sub-division of a tract.
# 
# We'll accept these flaws because it allows us to derive decent estimates given the data that we have access to. However, there are [resources](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4134912/) in the literature that may provide a "smarter" way of interpolating.
# 
# Ignoring these issues, for each SECTOR we'll do the following:
# - Using `geopandas.overlay`, we'll find the census tract [intersections](http://geopandas.org/set_operations.html)
# - We'll calculate the areal fraction of each of the intersected tracts to use as weighting factors for the intersected tract populations:
#     - `areal_fraction = intersected_tract_area/original_census_tract_area`
#     - For example, let's assume a census tract, 48001950100, intersects with a police SECTOR.
#         - `tract_48001950100_area = 100`
#         - `intersected_tract_48001950100_area = 40`
#         - -->`areal_fraction_interesected_tract_48001950100 = 0.4`
# - Therefore, the population of the intersected tract will be the `areal_fraction*census_tract_population`, where `census_tract_population` is available for each race
# - Finally, all of these newly calculated intersected tract populations for each race can be added up for each SECTOR
#     - We can also get the SECTOR racial population breakdown as percentages by dividing each racial population by the sum of the total population for each SECTOR
#     
# Thus, this method accounts for both the **relative populations within all the census tracts**, as well as the **division of some census tracts, split between multiple police SECTORs**.
#     
# To implement the above, we'll need quite a few `groupby`s and `merge`s, but the pandas/geopandas functions are fairly straightforward.
#     
# The UOF data gives us the SECTOR as `LOCATION_DISTRICT` (note: they're abbreviated). Therefore, all we have to do is use another `groupby` on the `LOCATION_DISTRICT` and apply a `value_counts` on the `SUBJECT_RACE` column to get the racial population breakdown of the UOF data.
# 
# Once we have both Census and UOF racial population breakdowns for all SECTORs, we can make some nice plots using Plotly to visualize potential bias.
# 
# **<font color='#d64541'>WARNING:</font>**
# I make a note in the code below, but it's worth repeating. It's important to note how the [census defines race and ethnicity](https://www.census.gov/mso/www/training/pdf/race-ethnicity-onepager.pdf) versus how the police departments are labeling race. Dept_37-00027 essentially has 4 labels within the `SUBJECT_RACE` category. Three of those labels (Black, White, Asian) are Race (or how the census defines race), and one of those labels is actually ethnicity (Hispanic). For example, my race is White, but my family comes from Mexico, so my ethnicity is Hispanic or Latino (i.e. I would tick both the "White" and the "Hispanic or Latino" boxes on the census form). Obviously, this complicates the analysis, but it also points to how complicated the [idea of Race as a label or category is](https://www.scientificamerican.com/article/race-is-a-social-construct-scientists-argue/). 
# 
# At the end of the day, we're interested in bias, which is related to perception, or how these police officers perceive the people they're arresting. I assume a police officer picks a label for the `SUBJECT_RACE` category, it's not up to the person to self-identify, as is the case in the census. So a disproportionate number of Use of Force incidents for a given `SUBJECT_RACE` will hopefully still indicate bias, since the `SUBJECT_RACE` is how police officers perceive the individuals they're arresting. Given these limitations, we'll use the "Hispanic or Latino" vs "Not Hispanic or Latino - RACE" categories, and build a dataset where we treat "Hispanic or Latino" as a race. While this is incorrect given how the census defines race and ethnicity, we're doing our best to align the census data with the police Use of Force data.
# 
# I've noticed some other Kernels that use the "One Race" columns for "White, Black, Asian, etc." (e.g. HC01_VC49) in addition to the Hispanic/Latino column (HC01_VC88). This is **<font color='#d64541'>incorrect</font>**. Because latino people can also be "White, Black, and Asian", this would be double counting. Instead, use the "Not Hispanic or Latino - RACE" (e.g. HC01_VC94 for "White").
# 
# [Jump to conclusion...](#conclusion)

# <a id='code'></a>

# In[ ]:


import numpy as np
import geopandas as gpd
import pandas as pd

from matplotlib import pyplot as plt
import plotly.offline as plotly
import plotly.graph_objs as go
import plotly.tools as tools

plotly.init_notebook_mode(connected=True)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


police_uof_path = "../input/data-science-for-good/cpe-data/Dept_37-00027/37-00027_UOF-P_2014-2016_prepped.csv"

# Use Pandas to read the "prepped" CSV, dropping the first row, which is just more headers
police_uof_df = pd.read_csv(police_uof_path).iloc[1:].reset_index(drop=True)


# In[ ]:


police_shp_path = "../input/data-science-for-good/cpe-data/Dept_37-00027/37-00027_Shapefiles/APD_DIST.shp"

# Use Geopandas to read the Shapefile
police_shp_gdf = gpd.read_file(police_shp_path)
police_shp_gdf.crs = {'init' :'esri:102739'}
police_shp_gdf = police_shp_gdf.to_crs(epsg='4326')


# In[ ]:


census_shp_path = "../input/cb-2017-48-tract-500k/cb_2017_48_tract_500k.shp"
census_tracts_gdf = gpd.read_file(census_shp_path)
census_tracts_gdf['GEOID'] = census_tracts_gdf['GEOID'].astype('int64')
census_tracts_gdf = census_tracts_gdf.to_crs(epsg='4326')


# In[ ]:


census_race_file_path = "../input/texas-14-5yr-dp05-racesexage/ACS_14_5YR_DP05.csv"


census_race_meta_file_path = "../input/texas-14-5yr-dp05-racesexage/ACS_14_5YR_DP05_metadata.csv"

census_race_df = pd.read_csv(census_race_file_path)
census_race_meta_df = pd.read_csv(census_race_meta_file_path,skiprows=2)
census_race_meta_df.columns = ['header','description']
meta_mask = ~census_race_meta_df['description'].str.contains('Margin of Error|SEX AND AGE',regex=True)
census_race_meta_df = census_race_meta_df[meta_mask] # Only keep "Percent" categories

census_columns_to_drop = ['HC02','HC04']
column_mask = census_race_df.columns.str.contains('|'.join(census_columns_to_drop), regex=True)
census_race_df = census_race_df.loc[:,census_race_df.columns[~column_mask]]
census_race_df = census_race_df.iloc[1:].reset_index(drop=True)

# Rename Census Tract ID column in ACS Poverty CSV to align with Census Tract Shapefile
census_race_df = census_race_df.rename(columns={'GEO.id2':'GEOID'})


# In[ ]:


print('Police UOF Race Labels:')
print(police_uof_df.SUBJECT_RACE.value_counts(dropna=False))
police_uof_df = police_uof_df.dropna(subset=['SUBJECT_RACE']).reset_index(drop=True)


# It's important to note how the [census defines race and ethnicity](https://www.census.gov/mso/www/training/pdf/race-ethnicity-onepager.pdf) versus how the police departments are labeling race. Note above, the police department essentially has 4 labels within the 'SUBJECT_RACE' category. Three of those labels (Black, White, Asian) are Race (or how the census defines race), and one of those labels is actually ethnicity (Hispanic). For example, my race is White, but my family comes from Mexico, so my ethnicity is Hispanic or Latino (i.e. I would tick both the "White" and the "Hispanic or Latino" boxes on the census form). Obviously, this complicates the analysis, but it also points to how complicated the [idea of Race as a label or category is](https://www.scientificamerican.com/article/race-is-a-social-construct-scientists-argue/).
# 
# At the end of the day, we're interested in bias, which is related to perception, or how these police officers perceive the people they're arresting. I assume a police officer picks a label for the "SUBJECT_RACE" category, it's not up to the person to self-identify, as is the case in the census. So a disproportionate number of Use of Force incidents for a given "SUBJECT_RACE" will hopefully still indicate bias, since the "SUBJECT_RACE" is how police officers perceive the individuals they're arresting. Given these limitations, we'll use the "Hispanic or Latino" vs "Not Hispanic or Latino - RACE" categories, and build a dataset where we treat "Hispanic or Latino" as a race. While this is incorrect given how the census defines race and ethnicity, we're doing our best to align the census data with the police Use of Force data.

# In[ ]:


race_perc_col_begin = 'Percent; HISPANIC OR LATINO AND RACE - Total population - '
race_est_col_begin = 'Estimate; HISPANIC OR LATINO AND RACE - Total population - '
race_col_end = ['Hispanic or Latino (of any race)',
                'Not Hispanic or Latino - White alone',
                'Not Hispanic or Latino - Black or African American alone',
                'Not Hispanic or Latino - American Indian and Alaska Native alone',
                'Not Hispanic or Latino - Asian alone',
                'Not Hispanic or Latino - Native Hawaiian and Other Pacific Islander alone',
                'Not Hispanic or Latino - Some other race alone',
                'Not Hispanic or Latino - Two or more races']
race_perc_cols = race_perc_col_begin+np.array(race_col_end,dtype=object)
race_est_cols = race_est_col_begin+np.array(race_col_end,dtype=object)


# In[ ]:


census_race_meta_df[census_race_meta_df['description'].isin(race_est_cols)].header.values


# In[ ]:


meta_perc_mask = census_race_meta_df['description'].isin(race_perc_cols)
meta_est_mask = census_race_meta_df['description'].isin(race_est_cols)
race_perc_headers = census_race_meta_df[meta_perc_mask].header.values
race_est_headers = census_race_meta_df[meta_est_mask].header.values
race_desc = np.array(['Hispanic','White','Black','Native American','Asian',
                      'Pacific Islander','Some Other Race','Two or more races']).astype(object)
race_perc_header_to_desc = dict(zip(race_perc_headers,race_desc+', Percent'))
race_est_header_to_desc = dict(zip(race_est_headers,race_desc+', Estimate'))
for item in list(zip(race_est_headers,race_desc)):
    print('{0}: {1}, Estimate'.format(item[0],item[1]))
print('')
for item in list(zip(race_perc_headers,race_desc)):
    print('{0}: {1}, Percent'.format(item[0],item[1]))


# In[ ]:


print(census_race_df.loc[0,race_perc_headers])
print('')
print('Sum: {0}'.format(census_race_df.loc[0,race_perc_headers].sum()))


# The above sums to 100 with some rounding error. These are the census column keys that we'll use going forward as our best proxy for the "race" labels in the police "Use of Force" data. But now we get back to the issue that the boundaries of the census tracts do not align nicely with the police precinct boundaries.

# In[ ]:


AX_LABEL_FONT_DICT = {'size':14}
AX_TITLE_FONT_DICT = {'size':16}

fig0,ax0 = plt.subplots()
(police_shp_gdf.dissolve(by='SECTOR').reset_index()
 .plot(ax=ax0,column='SECTOR',legend=True))
ax0.set_xlabel('Latitude (deg)',fontdict=AX_LABEL_FONT_DICT)
ax0.set_ylabel('Longitude (deg)',fontdict=AX_LABEL_FONT_DICT)
ax0.set_title('Dept_37-00027: Police Precincts (SECTORS)',
              fontdict=AX_TITLE_FONT_DICT)
leg0 = ax0.get_legend()
leg0.set_bbox_to_anchor((1.28, 1., 0., 0.))
leg0.set_title('SECTOR',prop={'size':12})
fig0.set_size_inches(7,7)


# In[ ]:


fig1,ax1 = plt.subplots()
census_tracts_gdf.plot(ax=ax1,color='#74b9ff',alpha=.4,edgecolor='white')
police_shp_gdf.plot(ax=ax1,column='SECTOR')
ax1.set_xlabel('Latitude (deg)',fontdict=AX_LABEL_FONT_DICT)
ax1.set_ylabel('Longitude (deg)',fontdict=AX_LABEL_FONT_DICT)
ax1.set_title('Dept_37-00027 and Texas Census Tracts',
              fontdict=AX_TITLE_FONT_DICT)
fig1.set_size_inches(8,11)


# The following lines fail in the Kaggle Kernels without `rtree`. I'm reading in a pickled GeoDataFrame from my local copy of this notebook. I suggest you run this notebook locally. 

# In[ ]:


police_sector_shp_gdf = police_shp_gdf.dissolve(by='SECTOR').reset_index()
#joined_df = gpd.overlay(police_sector_shp_gdf,census_tracts_gdf)
joined_df = pd.read_pickle("../input/cpe-joined-df/cpe_joined_df.pkl")


# The following cell is broken in the Kaggle Kernel. To run locally, uncomment all of the code below.

# In[ ]:


# fig2,ax2 = plt.subplots()
# joined_df.plot(ax=ax2,column='SECTOR',legend=True)
# ax2.set_xlabel('Latitude (deg)',fontdict=AX_LABEL_FONT_DICT)
# ax2.set_ylabel('Longitude (deg)',fontdict=AX_LABEL_FONT_DICT)
# ax2.set_title('Dept_37-00027: Police Precincts with Overlayed Census Tracts',
#               fontdict=AX_TITLE_FONT_DICT)
# leg2 = ax2.get_legend()
# leg2.set_bbox_to_anchor((1.28, 1., 0., 0.))
# leg2.set_title('SECTOR',prop={'size':12})
# fig2.set_size_inches(7,7)


# In[ ]:


def perc_tract_area(group, tracts_gdf):
    joined_area = group.area.values
    orig_tract_area = tracts_gdf.set_index('GEOID').loc[group['GEOID'].values,:].area.values
    perc_of_orig_tract = joined_area/orig_tract_area
    group['perc_of_orig_tract'] = perc_of_orig_tract
    return group


# In[ ]:


print(census_race_meta_df[census_race_meta_df['header'] == 'HC01_VC43'])


# In[ ]:


est_tot_pop_header = 'HC01_VC43'
joined_with_perc_area = (joined_df
                         .groupby('SECTOR')
                         .apply(perc_tract_area, census_tracts_gdf)
                         .sort_index())

joined_perc_area_and_pop = (joined_with_perc_area
                            .merge(census_race_df[['GEOID',est_tot_pop_header]],
                                   on='GEOID')
                            .sort_values('SECTOR')
                            .reset_index())

# Adjusting population based on percent area of census tract within police district
joined_perc_area_and_pop['pop_adj_by_area'] = (joined_perc_area_and_pop['perc_of_orig_tract']*
                                               joined_perc_area_and_pop[est_tot_pop_header])


# In[ ]:


def adj_pop_weight_factor(group):
    group['weight_factor'] = group['pop_adj_by_area']/group['pop_adj_by_area'].sum()
    return group


# In[ ]:


# Calculate areal_fractions to use as weight factors
joined_pop_weight_factor = (joined_perc_area_and_pop
                            .groupby('SECTOR')
                            .apply(adj_pop_weight_factor)
                            .drop('index',axis=1))

race_est_merge_headers = np.insert(race_est_headers,0,'GEOID')


joined_est_pop_weighted = (joined_pop_weight_factor
                           .merge(census_race_df[race_est_merge_headers],on='GEOID')
                           .sort_values('SECTOR').reset_index(drop=True))

# Use areal_fractions to "re-distribute" population of partial census tracts
est_pop_weighted = (joined_est_pop_weighted[race_est_headers]
                    .multiply(joined_est_pop_weighted['weight_factor'], 
                              axis="index"))


joined_est_pop_weighted = (pd.concat([joined_est_pop_weighted,
                                      est_pop_weighted.rename(columns=race_est_header_to_desc)],
                                      axis=1)
                           .drop(race_est_headers,axis=1))

# Sum all intersected_tract populations
race_est_by_pol_sector = (joined_est_pop_weighted
                          .groupby('SECTOR')[list(race_est_header_to_desc.values())]
                          .sum())
race_perc_by_pol_sector = (race_est_by_pol_sector
                           .divide(race_est_by_pol_sector.sum(axis=1),
                                   axis='index'))*100.
race_perc_by_pol_sector.columns = (race_perc_by_pol_sector
                                   .columns
                                   .str
                                   .replace('Estimate','Percent'))
race_perc_by_pol_sector.sort_index(axis=1,inplace=True)
race_perc_by_pol_sector


# In[ ]:


sectors_in_pol_shp = police_shp_gdf.SECTOR.value_counts(dropna=False).sort_index()
print(sectors_in_pol_shp)


# In[ ]:


print(police_uof_df.LOCATION_DISTRICT.value_counts().sort_index())


# Since there aren't any corresponding Sectors in the police precinct ShapeFile, we can drop "88", "-" and "NaN".

# In[ ]:


mask_missing_sectors = ~police_uof_df.LOCATION_DISTRICT.isin(['-','88',np.nan])
police_uof_df = police_uof_df[mask_missing_sectors].reset_index(drop=True)
sectors_in_pol_uof = police_uof_df.LOCATION_DISTRICT.value_counts().sort_index()
print(sectors_in_pol_uof)


# Much better...

# In[ ]:


sector_abbrev_dict = dict(zip(sectors_in_pol_uof.index,
                              sectors_in_pol_shp.index))
sector_abbrev_dict


# SECTORs in the police UOF data are abbreviated. For consistency, we replace the abbreviations with the proper names

# In[ ]:


police_uof_df.LOCATION_DISTRICT.replace(sector_abbrev_dict,inplace=True)


# In[ ]:


race_uof_est_by_pol_sector = (police_uof_df
                              .groupby('LOCATION_DISTRICT')['SUBJECT_RACE']
                              .value_counts()
                              .unstack())
race_uof_est_by_pol_sector.fillna(0,inplace=True)
race_uof_est_by_pol_sector.columns = race_uof_est_by_pol_sector.columns+', Estimate'
race_uof_est_by_pol_sector


# In[ ]:


race_uof_perc_by_pol_sector = (race_uof_est_by_pol_sector
                               .divide(race_uof_est_by_pol_sector.sum(axis=1),
                                       axis='index'))*100.
race_uof_perc_by_pol_sector.columns = (race_uof_perc_by_pol_sector
                                       .columns
                                       .str
                                       .replace('Estimate','Percent'))
race_uof_perc_by_pol_sector.drop('Unknown, Percent',axis=1,inplace=True)
race_uof_perc_by_pol_sector


# Interactive plotly magic below

# In[ ]:


missing_cols_mask = ~(race_perc_by_pol_sector
                      .columns
                      .isin(race_uof_perc_by_pol_sector.columns))
missing_cols = race_perc_by_pol_sector.columns[missing_cols_mask]
race_uof_perc_by_pol_sector = pd.concat([race_uof_perc_by_pol_sector,
                                         pd.DataFrame(columns=missing_cols)],
                                         axis=1,sort=False).fillna(0)
race_uof_perc_by_pol_sector.sort_index(axis=1,inplace=True)


# <a id='juicy'></a>

# In[ ]:


race_tick_labels = list(race_perc_by_pol_sector.columns.str.replace(', Percent',""))

for district_str in race_perc_by_pol_sector.index:
    race_breakdown_census = race_perc_by_pol_sector.loc[district_str,:]
    trace1 = go.Bar(
        y= race_breakdown_census.values,
        x= race_tick_labels,
        marker=dict(
            color='#34495e',
            line=dict(
                color='rgba(255, 255, 255, 0.0)',
                width=1),
        ),
        name='Census',
        orientation='v',
        showlegend = True
    )

    race_breakdown_uof = race_uof_perc_by_pol_sector.loc[district_str,:]
    trace2 = go.Bar(
        y= race_breakdown_uof.values,
        x= race_tick_labels,
        marker=dict(
            color='#ffa500',
            line=dict(
                color='rgba(255, 255, 255, 0.0)',
                width=1),
        ),
        name='Use of Force',
        orientation='v',
        showlegend = True
    )

    data = [trace1, trace2]
    layout = go.Layout(
        barmode='group',
        title = "Dept_37-00027: Racial Breakdown for Police Precinct (SECTOR) '{0}'".format(district_str)
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.iplot(fig)


# <a id='conclusion'></a>
# ## 3. Conclusion
# 
# This areal_fraction methodology can be used with other demographic information from the Census. It allows us to create snapshot visuals for departments, to help inform decision makers of how bias may or may not arise in different precincts. Care must be taken that we understand the categories and data definitions used by both the Census and the police departments. Fancy visuals can be distracting and misleading if we don't actually understand the underlying data. It is important to maintain open and clear communication with the police departments gathering and delivering this data. Without input from police officers and other stakeholders, interpreting the police Use of Force data can be very difficult. Feel free to leave comments if you have any ideas for how to improve this methodology. Best of luck to the rest of the participants.
# 
# [Back to top...](#top)

# In[ ]:




