#!/usr/bin/env python
# coding: utf-8

# # Kiva Crowdfunding - Adding a Financial Dimension to the MPI
# ***
# Kiva is an online crowdfunding platform to extend financial services to poor and financially excluded people around the world.
# This notebook series is my contribution to the Data Science for Good: Kiva Crowdfunding challenge. The objective is to help Kiva to better understand their borrowers and build more localized models to estimate the poverty levels in the regions where Kiva has active loans.
# 
# Kiva Crowdfunding notebook series:
#   - [Part I - Understanding Poverty]
#   - [Part II - Targeting Poverty at a National Level]
#   - [Part III - Targeting Poverty at a Subnational Level]
#   - [Part IV - Adding a Financial Dimension to the MPI]
#   - [Part V - Investigating Nightlight as a Poverty Indicator]
# 
# [Part I - Understanding Poverty]: https://www.kaggle.com/taniaj/kiva-crowdfunding-understanding-poverty
# [Part II - Targeting Poverty at a National Level]: https://www.kaggle.com/taniaj/kiva-crowdfunding-targeting-poverty-national
# [Part III - Targeting Poverty at a Subnational Level]: https://www.kaggle.com/taniaj/kiva-crowdfunding-targeting-poverty-sub-nat
# [Part IV - Adding a Financial Dimension to the MPI]: https://www.kaggle.com/taniaj/kiva-crowdfunding-adding-a-financial-dimension
# [Part V - Investigating Nightlight as a Poverty Indicator]: https://www.kaggle.com/taniaj/kiva-crowdfunding-investigating-nightlight
# 
# The series in broken down into five notebooks. The first notebook is an exploratory analysis of the data to get a feeling for what we are working with. The second notebook examines external datasets and looks at how MPI and other indicators can be used to get a better understanding of poverty levels of Kiva borrowers at a national level. The third notebook examines external data at a subnational level to see how Kiva can get MPI scores based on location at a more granular level than is currently available. The fourth notebook attepts to build a better poverty index at a subnational level by adding a financial dimension. The fifth notebook examines nightlight data as a poverty indicator.
# 
# This is the fourth notebook in the series and together with Part III, the most relevant to directly addressing the Kiva challenge.
# The notebook is broken down into two main sections. The first  uses the same DHS datasets used in Part III to combine a financial feature, that are not currently used as part of the OPHI MPI method, to add other dimensions to the MPI and build an index which could be more useful to Kiva. The second part looks at another external dataset, the Financial Inclusion Insights Survey by InterMedia, combined with DHS data to attempt to build a more representative index.
# 
# Note: The code has been written with a focus on understandability rather than optimization, although optimization is also a secondary aim.
# 
# ### Contents
# ***
#    1.  [Improved MPI (MPI2)](#improved_mpi) 
#    2.  [Data Gathering](#data_gathering)
#    3.  [Calculating MPI2](#calculating_mpi2)
#           * [Kenya](#kenya_mpi2)
#               * [Figure: Administration Level 1 MPI2](#fig_admin1_mpi2_kenya)
#               * [Figure:  Radar Plot of Decomposed Administrative Level 1 MPI2](#fig_radar_admin1_mpi2_kenya)         
#               * [Figure: Administration Level 2 MPI2](#fig_admin2_mpi2_kenya)
#    4.  [Assessing MPI2](#assessing_mpi2)
#           * [Rwanda](#rwanda_mpi2)
#               * [Figure: Administration Level 1 MPI2](#fig_admin1_mpi2_rwanda)
#               * [Figure:  Radar Plot of Decomposed Administrative Level 1 MPI2](#fig_radar_admin1_mpi2_rwanda)
#               * [Figure: Administration Level 2 MPI2](#fig_admin2_mpi2_rwanda)
#    
#    5.  [Improved MPI using Financial Inclusion Insights Data](#improved_mpi_fin_incl_insights)
#           * [Financial Inclusion Insights Surveys (2016 Surveys, available from Intermedia)](#fii_2016)
#               * [Overview of FII Data](#overview_fii_data)
#               * [Calculating the Financial Deprivation Indicator](#calculating_financial_deprivation_indicator)
#               * [Kenya](#ghana_mpi2_improved)
#                   * [Figure: Administration Level 2 MPI2 (Improved FDI)](fig_admin2_mpi2_improved_kenya)
#               * [Tanzania](#tanzania_mpi2_improved)
#           * [Financial Inclusion Insights Surveys (2014 Surveys, available from WorldBank)](#fii_2014)
#               * [Rwanda](#rwanda_mpi2_improved)
#               * [Ghana](#ghana_mpi2_improved)
#    6.  [Conclusion](#conclusion)
#    7.  [References](#references)

# ## 1. Improved MPI (MPI2) <a class="anchor" id="improved_mpi"/>
# ***
# In this section a new index will be built, reusing the dimensions from the existing MPI and adding in a fourth. Through research and work on this challenge, it has become obvious that one important aspect missing from the MPI for Kiva's purpose is a financial aspect. Kiva provides financial assistance and should have as part of any index used, a financial component, as here is where they can most directly help.
#  
# ## 2. Data Gathering <a class="anchor" id="data_gathering"/>
# ***
# The search for detailed (more detailed than country level) financial inclusion data has identified the best source currently available to be Financial Inclusion Insights Surveys by InterMedia, which currently offers detailed data for 12 developing countries. Unfortunately this is not enough to apply to most countries where Kiva has loans so it was decided to initially not include this data in the main model.

# ## 3. Calculating MPI2  <a class="anchor" id="calculating_mpi2"/>
# ***
# To build the new index (referred to as MPI2, for lack of a better idea), all the existing MPI indicators will be retained but reweighted to make room for the financial aspect of poverty. The financial deprivation indicator will be calculated using (unfortunately) the only directly relevant feature available from the DHS data - 'hv247' (Any member of the household has a bank account).
# 
# Initially, relevant data processing and MPI calculation methods that were developed in the Part III notebook are added here.
# For detailed explanation on how the methods that follow were developed, please refer to Part III of the series.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import seaborn as sns
from scipy.stats.mstats import gmean
import math
from scipy.stats.stats import pearsonr
#from geopandas.tools import sjoin
import folium
from folium.plugins import MarkerCluster
from folium import IFrame
import shapely
from shapely.geometry import Point, Polygon
import unicodedata
import pysal as ps
import geopandas as gpd
from mpl_toolkits.basemap import Basemap
import geojson

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(rc={"figure.figsize": (20,10), "axes.titlesize" : 18, "axes.labelsize" : 12, 
            "xtick.labelsize" : 14, "ytick.labelsize" : 14 }, 
        palette=sns.color_palette("OrRd_d", 20))

import warnings
warnings.filterwarnings('ignore')

get_ipython().system('cp ../input/images/rwanda_mpi2_decomposed_radar.png .')


# In[ ]:


# Functions to read in and preprocess data

def preprocess_dhs_data(country, household_file, househole_member_file, births_file, cluster_file):
    # Load original DHS data 
    # The following error occurrs if we do not set convert_categoricals=False: ValueError: Categorical categories must be unique
    household_dhs_df = pd.read_stata('../input/'+country+'-dhs-household/'+household_file, convert_categoricals=False)
    household_member_dhs_df = pd.read_stata('../input/'+country+'-dhs-household-member/'+househole_member_file, convert_categoricals=False)
    births_dhs_df = pd.read_stata('../input/'+country+'-dhs-births/'+births_file, convert_categoricals=False)
    dhs_cluster_df = pd.read_csv('../input/'+country+'-dhs-cluster/'+cluster_file)

    # Keep only relevant features from each dataset
    household_dhs_df = household_dhs_df[['hv001', 'hv002', 'hv009', 'hv010',  'hv011',  'hv012',  'hv014',  
                                         'hv024',  'hv025', 'hv027',
                                         'hv206','hv201','hv204','hv205','hv225', 'hv226','hv213',
                                         'hv207', 'hv208', 'hv243a', 'hv221',
                                        'hv210', 'hv211', 'hv212', 'hv243c', 'hv243d',
                                         'hv209', 'hv244', 'hv245', 'hv246', 
                                         'hv247']]
    household_member_dhs_df = household_member_dhs_df[['hv001', 'hv002', 'hc31', 'hc70', 'hc73', 'hc2', 'hc3','ha1', 
                                                       'ha40', 'hv105', 'hv108', 'hv121']]
    births_dhs_df = births_dhs_df[['v001', 'v002',  'b2', 'b3', 'b5', 'b7']]

    # Save the resulting dataframes
    household_dhs_df.to_csv(country+'_household_dhs.csv', index = False)
    household_member_dhs_df.to_csv(country+'_household_member_dhs.csv', index = False)
    births_dhs_df.to_csv(country+'_births_dhs.csv', index = False)

    # DHS Cluster data preprocessing
    # drop irrelevant columns
    dhs_cluster_df.drop(columns=['GPS_Dataset', 'DHSCC', 'DHSYEAR', 'SurveyID'], inplace=True)
    dhs_cluster_df = dhs_cluster_df[dhs_cluster_df.columns.drop(list(dhs_cluster_df.filter(regex='1985')))]
    dhs_cluster_df = dhs_cluster_df[dhs_cluster_df.columns.drop(list(dhs_cluster_df.filter(regex='1990')))]
    dhs_cluster_df = dhs_cluster_df[dhs_cluster_df.columns.drop(list(dhs_cluster_df.filter(regex='1995')))]
    dhs_cluster_df = dhs_cluster_df[dhs_cluster_df.columns.drop(list(dhs_cluster_df.filter(regex='2000')))]
    dhs_cluster_df = dhs_cluster_df[dhs_cluster_df.columns.drop(list(dhs_cluster_df.filter(regex='2005')))]
    dhs_cluster_df = dhs_cluster_df[dhs_cluster_df.columns.drop(list(dhs_cluster_df.filter(regex='UN_Population')))]
    dhs_cluster_df = dhs_cluster_df[dhs_cluster_df.columns.drop(list(dhs_cluster_df.filter(regex='SMOD')))]
    dhs_cluster_df = dhs_cluster_df[dhs_cluster_df.columns.drop(list(dhs_cluster_df.filter(regex='Slope')))]
    dhs_cluster_df = dhs_cluster_df[dhs_cluster_df.columns.drop(list(dhs_cluster_df.filter(regex='Temperature')))]
    dhs_cluster_df.to_csv(country+'_dhs_cluster.csv', index = False)

# Uncomment the line below to run pre-processing of original DHS files
#preprocess_dhs_data('kenya', 'KEHR71FL.DTA', 'KEPR71FL.DTA', 'KEBR71FL.DTA', 'KEGC71FL.csv')

# States-Provinces shapefile
states_provinces_gdf = gpd.read_file('../input/world-geo-data/ne_10m_admin_1_states_provinces.shp')
# Kiva subnational MPI dataset
mpi_subnational_df = pd.read_csv('../input/kiva-mpi-subnational-with-coordinates/mpi_subnational_coords.csv')

# This step is just to ensure we have matches where possible between the two datasets
#from string import punctuation
states_provinces_gdf['name'] = states_provinces_gdf['name'].str.replace('-',' ')
mpi_subnational_df['Sub-national region'] = mpi_subnational_df['Sub-national region'].str.replace('-',' ')

def read_data(country, household_path, household_member_path, births_path, dhs_cluster_path, dhs_geo_path, 
              admin1_geo_path, admin2_geo_path):
    global household_dhs_df
    global household_member_dhs_df
    global births_dhs_df
    global dhs_cluster_df
    global dhs_geo_gdf
    global admin1_geo_gdf
    global admin2_geo_gdf
    
    # Read in preprocessed DHS datasets
    household_dhs_df = pd.read_csv(household_path)
    household_member_dhs_df = pd.read_csv(household_member_path)
    births_dhs_df = pd.read_csv(births_path)
    dhs_cluster_df = pd.read_csv(dhs_cluster_path)
    # DHS shapefile
    dhs_geo_gdf = gpd.read_file(dhs_geo_path)
    # Admin1 boundaries shapefile
    admin1_geo_gdf = gpd.read_file(admin1_geo_path)
    # Admin2 boundaries shapefile
    admin2_geo_gdf = gpd.read_file(admin2_geo_path)


# In[ ]:


# Functions to process DHS data at raw feature level

# Determine drinking water deprivation
clean_water_source = [10, 11, 12, 13, 20, 21, 30, 31, 41, 51, 71]
def determine_water_depriv(row):
    if row.hv201 in clean_water_source:
        if (row.hv204 != 996) & (row.hv204 >= 30):
            return 1
        else:
            return 0
    else:
        return 1

# Determine asset deprivation given information_asset, mobility_asset and livelihood_asset features
def determine_asset_depriv(row):
    if row.information_asset == 0:
        return 1
    if (row.mobility_asset == 0) & (row.livelihood_asset == 0):
        return 1
    return 0
    
def process_household_data(df):
    df.rename(columns={'hv009':'total_household_members'}, inplace=True)
    df['financial_depriv'] = np.where(df['hv247'] == 0, 1, 0) 
    df['electricity_depriv'] = np.where(df['hv206'] == 0, 1, 0)
    df['water_depriv'] = df.apply(determine_water_depriv, axis=1)
    improved_sanitation =  [10, 11, 12, 13, 14, 15, 21, 22, 41]
    df['sanitation_depriv'] = np.where((df.hv225 == 0) & (df['hv205'].isin(improved_sanitation)), 0, 1)
    df['cooking_fuel_depriv'] = np.where(df['hv226'].isin([6, 7, 8, 9, 10, 11, 95, 96]), 1, 0)
    df['floor_depriv'] = np.where(df['hv213'].isin([11, 12, 13, 96]), 1, 0) 
    df['information_asset'] =  np.where((df.hv207 == 1) | (df.hv208 == 1) | (df.hv243a == 1) | (df.hv221 == 1), 1, 0)
    df['mobility_asset'] =  np.where((df.hv210 == 1) | (df.hv211 == 1) | (df.hv212 == 1) | (df.hv243c == 1) | (df.hv243d == 1), 1, 0)
    df['livelihood_asset'] =  np.where((df.hv209 == 1) | (df.hv244 == 1) | (df.hv245 == 1) | (df.hv246 == 1), 1, 0)
    df['asset_depriv'] = df.apply(determine_asset_depriv, axis=1)
    return df

# Nutrition:
z_cutoff_malnutrition = -200 # Below -2 Std deviations is considered malnourished (UNDP documentation)
bmi_cutoff_malnutrition = 1850 # Cutoff according is 18.5 (UNDP documentation)

def process_malnutrition(row):
    if not math.isnan(row['hc31']):
        if (row['hv105'] < 5): # < 5 years old
            if(row['hc70'] <= z_cutoff_malnutrition): # use Ht/A Std deviations
                return 1
            else:
                return 0
    elif not math.isnan(row['ha1']):
        if (row['hv105'] >= 15) & (row['hv105'] <= 49) & (row['ha40'] <= bmi_cutoff_malnutrition): # use BMI for adults
            return 1
        else:
            return 0
    else:
        return np.nan
    
def process_household_member_data(df):
    df['malnutrition'] = df.apply(process_malnutrition, axis=1)
    df['child_not_in_school'] = np.where((df['hv105'] >= 7) & (df['hv105'] <= 14) & (df['hv121'] == 0), 1, 0)
    df['child_under_5'] = np.where(df['hv105'] < 5, 1, 0)
    df['woman_15_to_49'] = np.where((df['ha1'] >= 15) & (df['ha1'] <=49), 1, 0)
    aggregations = {
        'hv108':lambda x: x.ge(6).sum(), # count number in houseold with >= 6 years of school
        'malnutrition': 'sum',
        'child_under_5': 'max',
        'woman_15_to_49': 'max',
        'child_not_in_school': 'max'
    }
    summary_df = df.groupby(['hv001', 'hv002']).agg(aggregations).reset_index()
    summary_df['school_attainment_depriv'] = np.where(summary_df['hv108'] == 0, 1, 0)
    summary_df['school_attendance_depriv'] = np.where(summary_df['child_not_in_school'] == 0, 0, 1)
    return summary_df

five_year_threshold = 2009 # Since the survey year was 2014 
def child_mortailty(row):
    if (row.b5 == 0) & (row.b2+(row.b7/12) >= five_year_threshold):
        return 1
    else:
        return 0
    
def process_births_data(df):
    df['child_mortailty'] = df.apply(child_mortailty, axis=1)
    aggregations = {
        'child_mortailty': 'sum'
    }
    return df.groupby(['v001', 'v002']).agg(aggregations).reset_index()

def combine_datasets(household_df, household_member_df, births_df):
    print("Original DHS household dataset: ", household_df.shape)
    combined_df = household_df.merge(household_member_df)
    combined_df = combined_df.merge(births_df, how='left', left_on=['hv001', 'hv002'], right_on=['v001', 'v002'])
    print("Merged dataset: ", combined_df.shape)
    
    # drop irrelevant columns
    combined_df = combined_df[combined_df.columns.drop(list(combined_df.filter(regex='^hv2')))]
    combined_df = combined_df[combined_df.columns.drop(list(combined_df.filter(regex='^v0')))]
    return combined_df


# In[ ]:


# MPI Calculation function and function to filter out eligible households

def calculate_deprivations(df, dhs_cluster_df, mp_threshold):
    # Calculate headcount ratio and poverty intensity
    df['headcount_poor'] =  np.where(df['total_of_weighted_deprivations'] >= mp_threshold, df['total_household_members'], 0)
    df['total_poverty_intensity'] = df['headcount_poor']*df['total_of_weighted_deprivations']

    # Format the DHSID to get just the number part for matching with hv001
    dhs_cluster_df['DHSID_num'] = dhs_cluster_df['DHSID'].str[6:].str.lstrip('0').astype(int)
    
    # Merge dhs_cluster with dhs_geo
    print("Original dhs_cluster_df dataset: ", dhs_cluster_df.shape)
    dhs_cluster_df = dhs_cluster_df.merge(dhs_geo_gdf[['DHSID', 'ADM1NAME', 'LATNUM', 'LONGNUM']], left_on=['DHSID'], right_on=['DHSID'], suffixes=('', '_y'))
    dhs_cluster_df = dhs_cluster_df[dhs_cluster_df.columns.drop(list(dhs_cluster_df.filter(regex='_y')))]
    print("Merged dhs_cluster_df dataset: ", dhs_cluster_df.shape)

    # Merge combined_df with dhs_cluster data to get county information (name)
    df = df.merge(dhs_cluster_df[['DHSID_num', 'ADM1NAME', 'LATNUM', 'LONGNUM']], left_on=['hv001'], right_on=['DHSID_num'])
    print("Merged df dataset: ", df.shape)
    return df

# Aggregate to specifed level, COUNTY level by default
def aggregate_admin_level(df, level='ADM1NAME', col='mpi_county'):
    aggregations = {
        'headcount_poor': 'sum',
        'total_household_members': 'sum',
        'total_poverty_intensity': 'sum'
    }
    df = df.groupby([level]).agg(aggregations).reset_index()

    # Calculate MPI at the required aggregation level
    df['headcount_ratio'] = df['headcount_poor']/df['total_household_members']
    df['poverty_intensity'] = df['total_poverty_intensity']/df['headcount_poor']
    df[col] = df['headcount_ratio'] * df['poverty_intensity']
    return df

def get_combined_data_for_eligible_households():
    global household_dhs_df
    global household_member_dhs_df
    global births_dhs_df
    
    # Process DHS data to get individual indicators
    household_dhs_df = process_household_data(household_dhs_df)
    household_member_dhs_summary_df = process_household_member_data(household_member_dhs_df)
    births_dhs_summary_df = process_births_data(births_dhs_df)
    combined_df = combine_datasets(household_dhs_df, household_member_dhs_summary_df, births_dhs_summary_df)

    # remove households with missing indicators
    print("Combined DHS Dataset: ", combined_df.shape)
    combined_df.dropna(inplace=True)
    print("Dataset after removing households with missing indicators: ", combined_df.shape)

    # remove ineligible households
    eligible_df = combined_df[(combined_df['woman_15_to_49'] != 0) | (combined_df['child_under_5'] != 0)]
    print("Dataset after removing ineligible households: ", eligible_df.shape)
    return eligible_df

def calculate_total_of_weighted_depriv(row):
    edu_ind_weight = 1/6
    health_ind_weight = 1/6
    liv_ind_weight = 1/18
    return (row.school_attainment_depriv*edu_ind_weight) + (row.school_attendance_depriv*edu_ind_weight) + (row.malnutrition*health_ind_weight) + (row.child_mortailty*health_ind_weight) + (row.electricity_depriv*liv_ind_weight) + (row.water_depriv*liv_ind_weight) + (row.sanitation_depriv*liv_ind_weight) + (row.cooking_fuel_depriv*liv_ind_weight) + (row.floor_depriv*liv_ind_weight) + (row.asset_depriv*liv_ind_weight)

# Function to run the whole process
# Note: The lines where sjoin is used are commented out in order to run on Kaggle servers. The data has been preprocessed locally,
# and read in when running on Kaggle. To run full sjoin steps, simple uncomment the lines.
def calculate_mpi(country, admin1_geo, admin1_col, admin1_mpi_col, admin2_geo=gpd.GeoDataFrame(), admin2_col='', admin2_mpi_col='', admin3_geo=gpd.GeoDataFrame(), admin3_col='', admin3_mpi_col=''):
    global household_dhs_df
    global household_member_dhs_df
    global births_dhs_df
    global dhs_mpi_df
    # Create them in case they are not produced
    admin2_dhs_mpi_df = pd.DataFrame()
    admin3_dhs_mpi_df = pd.DataFrame()
    
    # delete after debugging
    global dhs_mpi_joined_gdf

    eligible_df = get_combined_data_for_eligible_households()

    # calclate total weighted deprivations
    eligible_df['total_of_weighted_deprivations'] = eligible_df.apply(calculate_total_of_weighted_depriv, axis=1)

    # calculate MPI. mp_threshold is 0.333 because this is the cutoff for being considered multi-dimensionally poor 
    # (poor in more than one dimension, since there are 3 dimensions, this is 1/3)
    dhs_mpi_df = calculate_deprivations(eligible_df, dhs_cluster_df, 0.333)

    # Spatially join to admin1 boundaries
    #dhs_mpi_gdf = convert_to_geodataframe_with_lat_long(dhs_mpi_df, 'LONGNUM', 'LATNUM')
    #dhs_mpi_joined_gdf = gpd.sjoin(dhs_mpi_gdf, admin1_geo, op='within')
    #print("Dataset spatially joined with admin level 1 geodata: ", dhs_mpi_joined_gdf.shape)   
    #dhs_mpi_joined_gdf.to_csv(country+'_dhs_mpi_admin1_sjoin.csv', index = False)
    dhs_mpi_joined_gdf = pd.read_csv('../input/'+country.lower()+'-preprocessed/'+country+'_dhs_mpi_admin1_sjoin.csv')
    
    # Aggregate to admin1 (Province) level
    admin1_dhs_mpi_df = aggregate_admin_level(dhs_mpi_joined_gdf, level=admin1_col, col=admin1_mpi_col)
    print("Dataset aggregated to admin level 1: ", admin1_dhs_mpi_df.shape)
    
    # Ensure we are using title case for names (this is inconsistent in some country's datasets)
    admin1_dhs_mpi_df[admin1_col] = admin1_dhs_mpi_df[admin1_col].str.title()
    
    if not admin2_geo.empty:
        # Spatially join to admin2 boundaries
        #dhs_mpi_joined_gdf = gpd.sjoin(dhs_mpi_gdf, admin2_geo, op='within')
        #print("Dataset spatially joined with admin level 2 geodata: ", dhs_mpi_joined_gdf.shape)
        #dhs_mpi_joined_gdf.to_csv(country+'_dhs_mpi_admin2_sjoin.csv', index = False)
        dhs_mpi_joined_gdf = pd.read_csv('../input/'+country.lower()+'-preprocessed/'+country+'_dhs_mpi_admin2_sjoin.csv')
    if admin2_col:
        # Aggregate to admin2 (County) level
        admin2_dhs_mpi_df = aggregate_admin_level(dhs_mpi_joined_gdf, level=admin2_col, col=admin2_mpi_col)
        print("Dataset aggregated to admin level 2: ", admin2_dhs_mpi_df.shape)
    
    if not admin3_geo.empty:
        # Spatially join to admin3 boundaries
        #dhs_mpi_joined_gdf = gpd.sjoin(dhs_mpi_gdf, admin3_geo, op='within')
        #print("Dataset spatially joined with admin level 3 geodata: ", dhs_mpi_joined_gdf.shape)
        #dhs_mpi_joined_gdf.to_csv(country+'_dhs_mpi_admin3_sjoin.csv', index = False)
        dhs_mpi_joined_gdf = pd.read_csv('../input/'+country.lower()+'-preprocessed/'+country+'_dhs_mpi_admin3_sjoin.csv')
    if admin3_col:
        # Aggregate to admin3 level
        admin3_dhs_mpi_df = aggregate_admin_level(dhs_mpi_joined_gdf, level=admin3_col, col=admin3_mpi_col)
        print("Dataset aggregated to admin level 3: ", admin3_dhs_mpi_df.shape)

    return admin1_dhs_mpi_df, admin2_dhs_mpi_df, admin3_dhs_mpi_df


# In[ ]:


# Geometry and joining functions

# Function to combine MPI subnational scores with geometry
def get_mpi_subnational_gdf(mpi_subnational_df, states_provinces_gdf, country):
    # Keep just country data
    states_provinces_gdf = states_provinces_gdf[states_provinces_gdf['admin'] == country]
    mpi_subnational_df = mpi_subnational_df[mpi_subnational_df['Country'] == country]

    print("Country states_provinces_gdf dataset: ", states_provinces_gdf.shape)
    print("Country mpi_subnational_df dataset: ", mpi_subnational_df.shape)
    states_provinces_gdf.drop_duplicates(subset='woe_label', keep="last", inplace=True)
    print("Cleaned states_provinces_gdf dataset: ", states_provinces_gdf.shape)

    mpi_subnational_df = mpi_subnational_df[mpi_subnational_df['Country'] == country]
    mpi_subnational_df = mpi_subnational_df.merge(states_provinces_gdf, left_on='Sub-national region', right_on='name')
    print("Merged mpi_subnational_gdf dataset (with states_provinces_gdf): ", mpi_subnational_df.shape)
    return mpi_subnational_df

# Define some geo conversion functions
# Spatially join to counties
def convert_to_geodataframe_with_lat_long(df, lon, lat):
    df['geometry'] = df.apply(lambda row: Point(row[lon], row[lat]), axis=1)
    gdf = gpd.GeoDataFrame( df, geometry='geometry')
    gdf.crs = {"init":'epsg:4326'}
    return gdf

def convert_to_geodataframe_with_geometry(df, geometry):
    gdf = gpd.GeoDataFrame( df, geometry='geometry')
    gdf.crs = {"init":'epsg:4326'}
    return gdf

# Replace polygons with simple ones
def replace_geometry(gdf, gdf_simple_path):
    gdf_simple = gpd.read_file(gdf_simple_path)
    gdf['geometry'] = gdf_simple['geometry']
    
def get_geo_gdf(country):
    return states_provinces_gdf[states_provinces_gdf['geonunit'] == country]

def create_map(geo_gdf, data, key_on, key_col, feature, fill_color, lat, long, zoom, threshold_scale):
    geojson = geo_gdf.to_json()
    country_map = folium.Map([lat, long], zoom_start = zoom)
    country_map.choropleth(
        geo_data=geojson,
        name=feature+' choropleth',
        key_on=key_on,
        fill_color=fill_color,
        data=data,
        columns=[key_col, feature],
        threshold_scale=threshold_scale,
        legend_name= feature+' per Province'
    )
    return country_map


# To build the MPI2, a second version of the calculate_total_of_weighted_depriv function is added using new weightings and the new financial deprivation indicator.

# In[ ]:


edu_ind_weight_2 = 1/8     # 1/6 * 3/4
health_ind_weight_2 = 1/8  # 1/6 * 3/4
liv_ind_weight_2 = 1/24    # 1/18 * 3/4
fin_ind_weight_2 = 1/4
 
def calculate_total_of_weighted_depriv_2(row):
    return (row.school_attainment_depriv*edu_ind_weight_2) + (row.school_attendance_depriv*edu_ind_weight_2) + (row.malnutrition*health_ind_weight_2) + (row.child_mortailty*health_ind_weight_2) + (row.electricity_depriv*liv_ind_weight_2) + (row.water_depriv*liv_ind_weight_2) + (row.sanitation_depriv*liv_ind_weight_2) + (row.cooking_fuel_depriv*liv_ind_weight_2) + (row.floor_depriv*liv_ind_weight_2) + (row.asset_depriv*liv_ind_weight_2) + (row.financial_depriv*fin_ind_weight_2)


# The MPI calculation functions are updated to call the calculate_deprivations function with mp_threshold of 1/4, as there are now four dimensions of poverty in the model. It is also updated to call the new weighted deprivation calculation.
# 
# As with the previous MPI calculation function, sjoin functionality has been commented out and the results from a local execution uploaded and read in due to Kaggle servers not supporting it.

# In[ ]:


def calculate_mpi_2(country, admin1_geo, admin1_col, admin1_mpi2_col, admin2_geo=gpd.GeoDataFrame(), admin2_col='', admin2_mpi2_col=''):
    global household_dhs_df
    global household_member_dhs_df
    global births_dhs_df
    global dhs_mpi2_df
    
    eligible_df = get_combined_data_for_eligible_households()
    
    # calclate total weighted deprivations
    eligible_df['total_of_weighted_deprivations'] = eligible_df.apply(calculate_total_of_weighted_depriv_2, axis=1)

    # calculate MPI
    dhs_mpi2_df = calculate_deprivations(eligible_df, dhs_cluster_df, 0.25)

    # Spatially join to admin1 boundaries
    #dhs_mpi2_gdf = convert_to_geodataframe_with_lat_long(dhs_mpi2_df, 'LONGNUM', 'LATNUM')
    #dhs_admin1_mpi2_gdf = gpd.sjoin(dhs_mpi2_gdf, admin1_geo, op='within')
    #print("Dataset spatially joined with admin level 1 geodata: ", dhs_admin1_mpi2_gdf.shape)   
    #dhs_admin1_mpi2_gdf.to_csv(country+'_dhs_mpi2_admin1_sjoin.csv', index = False)
    dhs_admin1_mpi2_gdf = pd.read_csv('../input/'+country.lower()+'-preprocessed/'+country+'_dhs_mpi2_admin1_sjoin.csv')
    
    # Aggregate to admin1 (Province) level
    admin1_dhs_mpi2_df = aggregate_admin_level(dhs_admin1_mpi2_gdf, level=admin1_col, col=admin1_mpi2_col)
    print("Dataset aggregated to admin level 1: ", admin1_dhs_mpi2_df.shape)
    
    # Ensure we are using title case for names (this is inconsistent in some country's datasets)
    admin1_dhs_mpi2_df[admin1_col] = admin1_dhs_mpi2_df[admin1_col].str.title()
    
    if not admin2_geo.empty:
        # Spatially join to admin2 boundaries
        #dhs_admin2_mpi2_gdf = gpd.sjoin(dhs_mpi2_gdf, admin2_geo, op='within')
        #print("Dataset spatially joined with admin level 2 geodata: ", dhs_admin2_mpi2_gdf.shape)
        #dhs_admin2_mpi2_gdf.to_csv(country+'_dhs_mpi2_admin2_sjoin.csv', index = False)
        dhs_admin2_mpi2_gdf = pd.read_csv('../input/'+country.lower()+'-preprocessed/'+country+'_dhs_mpi2_admin2_sjoin.csv')
    if admin2_col:
        # Aggregate to admin2 (County) level
        admin2_dhs_mpi2_df = aggregate_admin_level(dhs_admin2_mpi2_gdf, level=admin2_col, col=admin2_mpi2_col)
        print("Dataset aggregated to admin level 2: ", admin2_dhs_mpi2_df.shape)

    return admin1_dhs_mpi2_df, admin2_dhs_mpi2_df, dhs_admin1_mpi2_gdf, dhs_admin2_mpi2_gdf


# ### Kenya <a class="anchor" id="kenya_mpi2"/>
# ***
# As with the previous MPI calculation (Part III notebook), the new model is tested out initially on Kenya.
# 
# The MPI calculation function developed in Part III is called to get the calculated MPI as presented in Part III. The MPI2 calcualtion function is then called to get the new MPI and the two scores are compared across administrative levels 1 and 2.

# In[ ]:


read_data('kenya', 
          '../input/kenya-preprocessed/kenya_household_dhs.csv',
          '../input/kenya-preprocessed/kenya_household_member_dhs.csv',
          '../input/kenya-preprocessed/kenya_births_dhs.csv',
          '../input/kenya-preprocessed/kenya_dhs_cluster.csv',
          '../input/kenya-preprocessed/KEGE71FL.shp', 
          '../input/kenya-humdata-admin-geo/Kenya_admin_2014_WGS84.shp', 
          '../input/kenya-humdata-admin-geo/KEN_Adm2.shp')

# Replace polygons with simple ones
replace_geometry(admin1_geo_gdf, '../input/kenya-humdata-admin-geo/Kenya_admin_2014_WGS84_simple.shp')

# Run the initial MPI calc again so that we can do comparisons
kenya_admin1_mpi_df, kenya_admin2_mpi_df, kenya_admin3_mpi_df = calculate_mpi('Kenya', admin1_geo_gdf, 'Province', 'mpi_admin1', 
        admin2_geo=admin2_geo_gdf, admin2_col='ADM1NAME', admin2_mpi_col='mpi_admin2', 
        admin3_col='Adm2Name', admin3_mpi_col='mpi_admin3')


# In[ ]:


# Merge 
kenya_mpi_subnational_gdf = get_mpi_subnational_gdf(mpi_subnational_df, states_provinces_gdf, 'Kenya')
kenya_admin1_mpi_merged_df = kenya_admin1_mpi_df.merge(kenya_mpi_subnational_gdf[['Sub-national region', 'MPI Regional']],
                                                left_on=['Province'], right_on=['Sub-national region'])
print("Dataset after merge with OPHI MPI data: ", kenya_admin1_mpi_merged_df.shape)


# In[ ]:


# Run MPI2 calc
kenya_admin1_mpi2_df, kenya_admin2_mpi2_df, kenya_admin1_mpi2_gdf, kenya_admin2_mpi2_gdf = calculate_mpi_2('Kenya', admin1_geo_gdf, 'Province', 'mpi2_admin1', 
        admin2_geo=admin2_geo_gdf, admin2_col='ADM1NAME', admin2_mpi2_col='mpi2_admin2')


# #### Figure: Administration Level 1 MPI2 - Kenya <a class="anchor" id="fig_admin1_mpi2_kenya"/>

# In[ ]:


kenya_mpi_threshold_scale = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # Define MPI scale for Kenya
kenya_geo_gdf = get_geo_gdf('Kenya')
create_map(kenya_geo_gdf, kenya_admin1_mpi2_df, 'feature.properties.name', 'Province', 'mpi2_admin1', 'YlOrRd', 0.0236, 37.9062, 6, kenya_mpi_threshold_scale)


# In[ ]:


# Merge 
country_mpi2_subnational_gdf = get_mpi_subnational_gdf(mpi_subnational_df, states_provinces_gdf, 'Kenya')
kenya_admin1_mpi2_merged_df = kenya_admin1_mpi2_df.merge(country_mpi2_subnational_gdf[['Sub-national region', 'MPI Regional']],
                                                left_on=['Province'], right_on=['Sub-national region'])
print("Dataset after merge with OPHI MPI data: ", kenya_admin1_mpi2_merged_df.shape)
    
# Check Correlation at admin1 level
print("Correlation, p-value: ", pearsonr(kenya_admin1_mpi2_merged_df.loc[:, 'mpi2_admin1'], kenya_admin1_mpi2_merged_df.loc[:, 'MPI Regional']))
sns.regplot(x="MPI Regional", y='mpi2_admin1', data=kenya_admin1_mpi2_merged_df)


# The result is surprising and a little disappointing because the high correlation between the MPI2 and OPHI MPI means that the financial component that was added does not really change much about the poverty estimate between regions for Kenya. It does look like the MPI is slightly higher for most regions as a result of the inclusion of financial deprivation in the calculation (as seen in the correlation  plot). Lets have a look at some radar plots to examine the calculated financial deprivation more closely.

# In[ ]:


# Aggregate to province
def aggregate_individual_indicators(df, region_col):
    aggregations = {
        'headcount_poor': 'sum',
        'total_household_members': 'sum',
        'total_poverty_intensity': 'sum',
        'electricity_depriv': 'sum',
        'water_depriv': 'sum',
        'sanitation_depriv': 'sum',
        'cooking_fuel_depriv': 'sum',
        'floor_depriv': 'sum',
        'asset_depriv': 'sum',
        'malnutrition': 'sum',
        'child_mortailty': 'sum',
        'school_attainment_depriv': 'sum',
        'school_attendance_depriv': 'sum',
        'financial_depriv' : 'sum'
    }
    return df.groupby([region_col]).agg(aggregations).reset_index()
    
province_dhs_mpi2_decomp_df = aggregate_individual_indicators(kenya_admin1_mpi2_gdf, 'Province' )

# Calculate deprivation raw headcount ratios
def get_headcount_ratios_for_all_indicators(df):
    df['electricity_depriv_ratio'] = df['electricity_depriv']/df['total_household_members']
    df['water_depriv_ratio'] = df['water_depriv']/df['total_household_members']
    df['sanitation_depriv_ratio'] = df['sanitation_depriv']/df['total_household_members']
    df['cooking_fuel_depriv_ratio'] = df['cooking_fuel_depriv']/df['total_household_members']
    df['floor_depriv_ratio'] = df['floor_depriv']/df['total_household_members']
    df['asset_depriv_ratio'] = df['asset_depriv']/df['total_household_members']
    df['malnutrition_ratio'] = df['malnutrition']/df['total_household_members']
    df['child_mortailty'] = df['child_mortailty']/df['total_household_members']
    df['school_attainment_depriv_ratio'] = df['school_attainment_depriv']/df['total_household_members']
    df['school_attendance_depriv_ratio'] = df['school_attendance_depriv']/df['total_household_members']
    df['financial_depriv_ratio'] = df['financial_depriv']/df['total_household_members']

    df['headcount_ratio'] = df['headcount_poor']/df['total_household_members']
    df['poverty_intensity'] = df['total_poverty_intensity']/df['headcount_poor']
    df['mpi'] = df['headcount_ratio'] * df['poverty_intensity']
    return df

province_mpi2_decomp_df = get_headcount_ratios_for_all_indicators(province_dhs_mpi2_decomp_df)


# In[ ]:


def radar_plot(df, columns, labels, title_col, num_rows, num_cols):
    fig = plt.figure(figsize=(5*num_cols,5*num_rows))
    for i, (name, row) in enumerate(df.iterrows()):
        stats=df.loc[i, columns].values

        # Create a color palette:
        palette = plt.cm.get_cmap("Set2", num_rows*num_cols)

        angles=np.linspace(0, 2*np.pi, len(columns), endpoint=False)
        # close the plot
        stats=np.concatenate((stats,[stats[0]]))
        angles=np.concatenate((angles,[angles[0]]))

        ax = plt.subplot(num_rows, num_cols, i+1, polar=True)
        ax.plot(angles, stats, linewidth=2, linestyle='solid', color=palette(i))
        ax.fill(angles, stats, color=palette(i), alpha=0.6)
        ax.set_theta_offset(1.85) # Change the rotation so that labels don't overlap
        ax.set_rmax(0.2)
        ax.set_rticks([0.05, 0.1, 0.15, 0.2])  # less radial ticks
        ax.set_thetagrids(angles * 180/np.pi, labels)
        ax.set_title(row[title_col], color=palette(i))
        ax.grid(True)
        
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=2.0, rect=[0, 0, 0.95, 0.95])
    plt.show()


# #### Figure: Radar Plot of Decomposed Administrative Level 1 MPI2  <a class="anchor" id="fig_radar_admin1_mpi2_kenya"/>

# In[ ]:


columns=np.array(['electricity_depriv_ratio', 'water_depriv_ratio', 'sanitation_depriv_ratio', 'cooking_fuel_depriv_ratio', 'floor_depriv_ratio',
                 'asset_depriv_ratio','malnutrition_ratio','child_mortailty','school_attainment_depriv_ratio','school_attendance_depriv_ratio', 
                 'financial_depriv_ratio'])
labels=np.array(['elect', 'water', 'sanit', 'cook', 'flr',
                 'asset','maln','mort','sch1','sch2', 'fin'])
radar_plot(province_mpi2_decomp_df, columns, labels, 'Province', 3, 3)


# The radar plots seem to indicate that the financial deprivation idicator indeed doesn't seem to vary hugely between provinces. The only province with a noticeably lower financial deprivation than the rest is, not surprisingly, Nairobi, the capital of Kenya.
# 
# Further investigation is required to determine if including this financial deprivation indicator actually improves the index for Kiva's purposes or not. Next the impact at the county level will be looked at.

# In[ ]:


# Correlation between calculated MPI1 and MPI2
print("Correlation, p-value: ", pearsonr(kenya_admin2_mpi2_df.loc[:, 'mpi2_admin2'], kenya_admin2_mpi_df.loc[:, 'mpi_admin2']))
sns.regplot(x=kenya_admin2_mpi_df.mpi_admin2, y=kenya_admin2_mpi2_df.mpi2_admin2)


# #### Figure: Administration Level 2 MPI2 - Kenya <a class="anchor" id="fig_admin2_mpi2_kenya"/>

# In[ ]:


create_map(admin1_geo_gdf, kenya_admin2_mpi2_df, 'feature.properties.COUNTY', 'ADM1NAME', 'mpi2_admin2', 'YlOrRd', 0.0236, 37.9062, 6, kenya_mpi_threshold_scale)


# Adding the financial deprivation dimension doesn't seem to have altered much between regions for Kenya but it does look like it has raised the overall MPI for the country. 
# 
# More work is needed to look at the impact on other countries.

# ### Rwanda <a class="anchor" id="rwanda_mpi2"/>
# ***
# In this section, the MPI2 calculation developed previously is tested out on Rwanda data.

# In[ ]:


read_data('rwanda', 
          '../input/rwanda-preprocessed/rwanda_household_dhs.csv',
          '../input/rwanda-preprocessed/rwanda_household_member_dhs.csv',
          '../input/rwanda-preprocessed/rwanda_births_dhs.csv',
          '../input/rwanda-preprocessed/rwanda_dhs_cluster.csv',
          '../input/rwanda-preprocessed/RWGE72FL.shp', 
          '../input/rwanda-humdata-admin-geo/RWA_Admin2_2006_NISR.shp', 
          '../input/rwanda-humdata-admin-geo/RWA_Admin3_2006_NISR.shp')

# Simplify geometry. Seems to be necessary only for admin level 2 for Zimbabwe.
replace_geometry(admin2_geo_gdf, '../input/rwanda-humdata-admin-geo/RWA_Admin3_2006_NISR_simple.shp')

# Doing some manual recoding to get matches 
states_provinces_gdf.name.replace('Southern', 'South', inplace=True)
states_provinces_gdf.name.replace('Northern', 'North', inplace=True)
states_provinces_gdf.name.replace('Eastern', 'East', inplace=True)
states_provinces_gdf.name.replace('Western', 'West', inplace=True)

admin1_geo_gdf.PROVINCE.replace('SOUTHERN PROVINCE', 'South', inplace=True)
admin1_geo_gdf.PROVINCE.replace('NORTHERN PROVINCE', 'North', inplace=True)
admin1_geo_gdf.PROVINCE.replace('EASTERN PROVINCE', 'East', inplace=True)
admin1_geo_gdf.PROVINCE.replace('WESTERN PROVINCE', 'West', inplace=True)
admin1_geo_gdf.PROVINCE.replace('TOWN OF KIGALI', 'Kigali City', inplace=True)

admin2_geo_gdf['NOMDISTR'] = admin2_geo_gdf['NOMDISTR'].str.title()

# Run the initial MPI calc again so that we can do comparisons
rwanda_admin1_mpi_df, rwanda_admin2_mpi_df, rwanda_admin3_mpi_df = calculate_mpi('Rwanda', admin1_geo_gdf, 'ADM1NAME', 'mpi_admin1', 
                                     admin2_geo=admin2_geo_gdf, admin2_col='NOMDISTR', admin2_mpi_col='mpi_admin2')


# In[ ]:


# Run MPI2 calc
rwanda_admin1_mpi2_df, rwanda_admin2_mpi2_df, rwanda_admin1_mpi2_gdf, rwanda_admin2_mpi2_gdf = calculate_mpi_2('Rwanda', admin1_geo_gdf, 'ADM1NAME', 'mpi2_admin1', 
                                         admin2_geo=admin2_geo_gdf, admin2_col='NOMDISTR', admin2_mpi2_col='mpi2_admin2')


# #### Figure: Administration Level 1 MPI2 - Rwanda <a class="anchor" id="fig_admin1_mpi2_rwanda"/>

# In[ ]:


rwanda_mpi_threshold_scale = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] # Define MPI scale
rwanda_geo_gdf = get_geo_gdf('Rwanda')
create_map(rwanda_geo_gdf, rwanda_admin1_mpi2_df, 'feature.properties.name', 'ADM1NAME', 'mpi2_admin1', 'YlOrRd', -1.9403, 29.8739, 8, rwanda_mpi_threshold_scale)


# In[ ]:


# Merge 
country_mpi2_subnational_gdf = get_mpi_subnational_gdf(mpi_subnational_df, states_provinces_gdf, 'Rwanda')
rwanda_admin1_mpi2_merged_df = rwanda_admin1_mpi2_df.merge(country_mpi2_subnational_gdf[['Sub-national region', 'MPI Regional']],
                                                left_on=['ADM1NAME'], right_on=['Sub-national region'])
print("Dataset after merge with OPHI MPI data: ", rwanda_admin1_mpi2_merged_df.shape)
    
# Check Correlation at admin1 level
print("Correlation, p-value: ", pearsonr(rwanda_admin1_mpi2_merged_df.loc[:, 'mpi2_admin1'], rwanda_admin1_mpi2_merged_df.loc[:, 'MPI Regional']))
sns.regplot(x="MPI Regional", y='mpi2_admin1', data=rwanda_admin1_mpi2_merged_df)


# #### Figure: Radar Plot of Decomposed Administrative Level 1 MPI2  <a class="anchor" id="fig_radar_admin1_mpi2_rwanda"/>

# In[ ]:


province_mpi2_decomp_df = aggregate_individual_indicators(rwanda_admin1_mpi2_gdf, 'ADM1NAME' )
province_mpi2_decomp_df = get_headcount_ratios_for_all_indicators(province_mpi2_decomp_df)

columns=np.array(['electricity_depriv_ratio', 'water_depriv_ratio', 'sanitation_depriv_ratio', 'cooking_fuel_depriv_ratio', 'floor_depriv_ratio',
                 'asset_depriv_ratio','malnutrition_ratio','child_mortailty','school_attainment_depriv_ratio','school_attendance_depriv_ratio', 
                 'financial_depriv_ratio'])
labels=np.array(['elect', 'water', 'sanit', 'cook', 'flr',
                 'asset','maln','mort','sch1','sch2', 'fin'])
radar_plot(province_mpi2_decomp_df, columns, labels, 'ADM1NAME', 2, 3)


# The result of adding the financial dimension is the same for Rwanda - there is not much change between regions at the first administrative level although the overall score seems to have increased. The only area that looks like it has a significantly different (lower) financial deprivation is Kigali City.
# 
# Lets see what happens at the administrative level 2.

# In[ ]:


# Correlation between calculated MPI1 and MPI2
print("Correlation, p-value: ", pearsonr(rwanda_admin2_mpi2_df.loc[:, 'mpi2_admin2'], rwanda_admin2_mpi_df.loc[:, 'mpi_admin2']))
sns.regplot(x=rwanda_admin2_mpi_df.mpi_admin2, y=rwanda_admin2_mpi2_df.mpi2_admin2)


# #### Figure: Administration Level 2 MPI2 - Rwanda <a class="anchor" id="fig_admin2_mpi2_rwanda"/>

# In[ ]:


create_map(admin2_geo_gdf, rwanda_admin2_mpi2_df, 'feature.properties.NOMDISTR', 'NOMDISTR', 'mpi2_admin2', 'YlOrRd', -1.9403, 29.8739, 8, rwanda_mpi_threshold_scale)


# At the administrative level 2 there is a lower correlation between MPI2 and MPI, suggesting that the financial dimension may add more value when looking at this level.

# ## 2. Improved MPI using Financial Inclusion Insights Data <a class="anchor" id="improved_mpi_fin_incl_insights"/>
# ***
# ### Financial Inclusion Insights Surveys (2016 Surveys, available from Intermedia) 
# 
# The single feature identified in the DHS surveys related to financial inclusion is perhaps by itself not enough to give a clear picture of financial inclusion. Therefore, a search was done to see what other financial inclusion data exists for developing countries at a granular level. The search revealed the Financial Inclusion Insights (FII) Surveys done by Intermedia as part of their Financial Inclusion Program (http://finclusion.org/). The program covers 12 countries although the author was only able to obtain data for 8 (Intermedia 2016 Surveys - Uganda, Tanzania, Pakistan, Nigeria, Kenya, Indonesia, India and Bangladesh) through Intermedia. Data for 2 additional countries (Intermedia 2014 Surveys - Rwanda and Ghana) was found on the World Bank website. Why Intermedia does not make all 12 countries data available is not clear. However, it seemed like enough data to run some analysis on to determine whether it is possible to get a better financial indicator as a dimension in an MPI.
# 
# ### Overview of the FII Data <a class="anchor" id="overview_fii_data"/>
# ***
# The following relevant information was taken from Intermedia documentation of the 2016 surveys. Additional features are also present with summarising plain-text descriptions (such as registered_bank_full, registered_mm, etc. which wil also be used).
# 
#     The samples of the FII surveys cover the adult populations of the countries, not the entire populations. In most cases, this means the samples cover individuals ages 15 and over. This is important note when analyzing age-related data, which cannot be generalized to the population.
# 
#     Relevant features:
#         -  DG8a: How many adults and children do you have in the household? (99 for DK): Number of adults
#         -  DG8b: How many adults and children do you have in the household? (99 for DK): Number of children (boys) 
#         -  DG8c: How many adults and children do you have in the household? (99 for DK): Number of children (girls)
# 
#         **Savings**
#         -  FL6.4: Tell me, does the following apply to you? My savings is larger than my debt
#         -  FB22.1: Do you save with any of the following? Bank
#         -  FB22.2: Do you save with any of the following? Mobile money account or mobile money product 	
#         -  FB22.3: Do you save with any of the following? Savings account at a SACCO, a member-based organization (e.g., Mwalimu or workplace co-op)
#         -  FB22.4: Do you save with any of the following? Savings at a microfinance institution (organization which mostly lends to members in a group, e.g., KWFT, Faulu) 
# 
#         **Borrowing**
#         -  FB 3: Have you needed credit in the past year but could not access it?
# 
#         *For all questions, 1=Yes, 2=No

# ### Calculating the Financial Deprivation Indicator (FDI) <a class="anchor" id="calculating_financial_deprivation_indicator"/>
# ***
# In calculating a new financial deprivation dimension, the learnings from notebook II and III are used, namely:
# 1. The World Bank key indicators for financial inclusion:
#     - percentage of people (age 15+) who have an account
#     - percentage of people (age 15+) who have formal savings
#     - percentage of people (age 15+) who have access to formal borrowing
# 2. The OPHI MPI calcualtion basis that everyone in the household is considered deprived if the surveyed member is deprived.
# 3. The idea of multidimensional deprivation, where a household is multidimensionally poor when they are deprived in more than one indicator.
# 
# To calculate the total household members, DG8a through c are added together. 
# If at least one of FB22.1 through to FB22.4 are answered "Yes", the respondent and all in their household are considered to have formal savings.
# If FB3 is answered "Yes", the respondent and all in their household are considered to be deprived of borrowing.
# If the respondent has either "registered_bank_full" or "registered_mm" (mobile money), them and all in their household are considered to have a formal account.
# 
# These three areas of deprivation are then added together with equal weights (1/3 each) to determine whether the household is financially multidimensionally deprived/poor (ie: when they are deprived in more than one financial indicator).
# 
# These results are used as the basis to calculate a headcount of poor people and poverty intensity with respect to financial poverty. The final result is then added to the original (reweighted to 3/4) MPI with a weight of 1/4.

# In[ ]:


def preprocess_intermed_data(country, filepath):
    df = pd.read_csv(filepath, encoding='cp1252')
    df = df[['AA1','AA2','AA3','AA4', 'Stratum', 
             #'Latitude', 
               'age', 'DG8a', 'DG8b', 'DG8c', 'poverty','ppi_score','UR','AA7', 
               'access_phone_SIM', 'own_phone_SIM', 'registered_mm', 'nonregistered_mm',
               'registered_bank_full', 'nonregistered_bank_full', 
               'FB22_1','FB22_2','FB22_3','FB22_4', 'FB3']]
    df.rename(columns={'FB22_1':'bank_savings'}, inplace=True)
    df.rename(columns={'FB22_2':'mm_savings'}, inplace=True)
    df.rename(columns={'FB22_3':'other_reg_savings'}, inplace=True)
    df.rename(columns={'FB22_4':'micro_savings'}, inplace=True)
    df.rename(columns={'FB3':'couldnt_access_credit'}, inplace=True)
    df['total_household_members'] = df['DG8a']+df['DG8b']+df['DG8c']
    df.to_csv(country+'_fii_preprocessed.csv', index = False)

def process_fii_data(df):
    df['acct_depriv'] = np.where((df['registered_bank_full']==1) | (df['registered_mm']==1), 0, 1)
    df['saving_depriv'] = df[['bank_savings','mm_savings','other_reg_savings','micro_savings']].min(axis=1)
    df['saving_depriv'].replace(1, 0, inplace=True)
    df['saving_depriv'].replace(2, 1, inplace=True)
    df['borrowing_depriv'] = np.where(df['couldnt_access_credit']==1, 1, 0)
    # Calculate financial deprivation indicator
    # Attempting to keep the definition uniform, lets say that someone is financially deprived if they are deprived in more than one
    # financial indicator.
    df['financial_depriv'] = np.where(df['acct_depriv'] + df['saving_depriv'] + df['borrowing_depriv'] > 1, 1, 0)
    return df

def calculate_total_weighted_fin_depriv(row):
    fin_ind_weight = 1/3
    return (row.acct_depriv*fin_ind_weight) + (row.saving_depriv*fin_ind_weight) + (row.borrowing_depriv*fin_ind_weight) 

def calculate_fin_deprivations(df):
    fin_ind_weight = 1/3
    mp_threshold = 1/3

    # calclate total weighted deprivations
    df['total_of_weighted_deprivations'] = df.apply(calculate_total_weighted_fin_depriv, axis=1)
    # Calculate headcount poor and poverty intensity
    df['headcount_poor'] =  np.where(df['total_of_weighted_deprivations'] >= mp_threshold, df['total_household_members'], 0)
    df['total_poverty_intensity'] = df['headcount_poor']*df['total_of_weighted_deprivations']
    return df

def calculate_mpi_with_fin_dimension(df, mpi_col, fin_poverty_col):
    mpi_weight = 3/4
    fin_weight = 1/4
    df['mpi2'] = (df[mpi_col]*mpi_weight) + (df[fin_poverty_col]*fin_weight)
    return df
    
def calculate_mpi2_improved_fin_dim(mpi_df, fin_df, mpi_region_col, fin_region_col, mpi_col):
    fin_df = process_fii_data(fin_df)
    fin_df = calculate_fin_deprivations(fin_df)
    fin_summary_df = aggregate_admin_level(fin_df, level=fin_region_col, col='fin_poverty')
    print("Dataset mpi_df: ", mpi_df.shape)
    mpi_fin_df = mpi_df.merge(fin_summary_df[[fin_region_col, 'fin_poverty']], how='left', left_on=[mpi_region_col], right_on=[fin_region_col])
    print("Dataset mpi_df after merge with fin_df: ", mpi_fin_df.shape)
    mpi_fin_df = calculate_mpi_with_fin_dimension(mpi_fin_df, mpi_col, 'fin_poverty')
    return mpi_fin_df

def check_correlation(mpi_fin_df, mpi_col, fin_poverty_col, mpi2_col):
    # Check Correlation at region level
    print("MPI vs Fin Poverty correlation, p-value: ", pearsonr(mpi_fin_df.loc[:, fin_poverty_col], mpi_fin_df.loc[:, mpi_col]))
    sns.regplot(x=mpi_col, y=fin_poverty_col, data=mpi_fin_df)
    plt.figure()
    print("MPI vs MPI2 correlation, p-value: ", pearsonr(mpi_fin_df.loc[:, mpi2_col], mpi_fin_df.loc[:, mpi_col]))
    sns.regplot(x=mpi_col, y=mpi2_col, data=mpi_fin_df)
    plt.figure()


# ### Kenya <a class="anchor" id="kenya_mpi2_improved"/>
# ***
# Now the MPI2 with improved financial deprivation indicator will be calculated and analysed for Kenya.

# In[ ]:


#preprocess_intermed_data('kenya', '../input/financial-inclusion-insights/FII_ 2016_Kenya_Wave_4_Data.csv')


# In[ ]:


kenya_fin_df = pd.read_csv('../input/kenya-preprocessed/kenya_fii_preprocessed.csv',)

# Kenya-specific string processing
kenya_fin_df['Region'] = kenya_fin_df.Stratum.str.split('_').str[0]
kenya_fin_df['Region'] = kenya_fin_df['Region'].str.title()

# TODO: Update Region strings to match ADM1NAME
kenya_fin_df['Region'].replace('Muranga', "Murang'a", inplace=True)
kenya_fin_df['Region'].replace('Tharaka', "Tharaka-Nithi", inplace=True)

kenya_fin_df.sample(5)


# In[ ]:


kenya_mpi_fin_df = calculate_mpi2_improved_fin_dim(kenya_admin2_mpi_df, kenya_fin_df, 'ADM1NAME', 'Region', 'mpi_admin2')


# #### Figure: Administration Level 2 MPI2 (Improved FDI) - Kenya <a class="anchor" id="fig_admin2_mpi2_improved_kenya"/>

# In[ ]:


kenya_admin1_geo_gdf = gpd.read_file('../input/kenya-humdata-admin-geo/Kenya_admin_2014_WGS84.shp')
replace_geometry(kenya_admin1_geo_gdf, '../input/kenya-humdata-admin-geo/Kenya_admin_2014_WGS84_simple.shp')
create_map(kenya_admin1_geo_gdf, kenya_mpi_fin_df, 'feature.properties.COUNTY', 'ADM1NAME', 'mpi2', 'YlOrRd', 0.0236, 37.9062, 6, kenya_mpi_threshold_scale)


# In[ ]:


check_correlation(kenya_mpi_fin_df, 'mpi_admin2', 'fin_poverty', 'mpi2')


# This result is, at first look, a little strange. The first plot shows that there is only a moderate correlation between the financial poverty measure and the calculated MPI. The second plot shows a very strong correlation between MPI2 and the calculated MPI, even though a quarter of the value of MPI2 comes from the financial poverty measure. It should however, be kept in mind that correlation is not transitive.

# In[ ]:


plt.subplot(221).set_title("Kenya County MPI distribuion")
sns.distplot(kenya_mpi_fin_df.mpi_admin2, bins=30)

plt.subplot(222).set_title("Kenya County fin_poverty distribuion")
sns.distplot(kenya_mpi_fin_df.fin_poverty, bins=30)

plt.subplot(223).set_title("Kenya County MPI2 distribuion")
sns.distplot(kenya_mpi_fin_df.mpi2, bins=30)

plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=2.0, rect=[0, 0, 0.95, 0.95])


# In[ ]:


kenya_mpi_fin_df[['mpi_admin2', 'mpi2']].describe()


# The above histograms and summary statistics for MPI and MPI2 show that MPI has a slightly higher standard deviation than MPI2. Both distributions have a similar range of values but the MPI2 is slightly higher than MPI. 

# ### Tanzania  <a class="anchor" id="tanzania_mpi2_improved"/>
# ***
# Now that the improved financial dimension calculation has been developed using Kenya data, lets test it out on Tanzania. 

# In[ ]:


read_data('tanzania', 
          '../input/tanzania-preprocessed/tanzania_household_dhs.csv',
          '../input/tanzania-preprocessed/tanzania_household_member_dhs.csv',
          '../input/tanzania-preprocessed/tanzania_births_dhs.csv',
          '../input/tanzania-preprocessed/tanzania_dhs_cluster.csv',
          '../input/tanzania-preprocessed/TZGE7AFL.shp', 
          '../input/tanzania-humdata-admin-geo/tza_popa_adm1_regions_TNBS2012_OCHA.shp', 
          '../input/tanzania-humdata-admin-geo/tza_popa_adm2_districts_TNBS2012_OCHA.shp')


# In[ ]:


admin1_geo_gdf.sample()


# In[ ]:


# Run the initial MPI calc again so that we can do comparisons
tanzania_admin1_mpi_df, tanzania_admin2_mpi_df, tanzania_admin3_mpi_df = calculate_mpi('Tanzania', admin1_geo_gdf, 'REGION', 'mpi_admin1', 
        admin2_geo=admin2_geo_gdf, admin2_col='ADM1NAME', admin2_mpi_col='mpi_admin2')


# In[ ]:


# Run MPI2 calc
tanzania_admin1_mpi2_df, tanzania_admin2_mpi2_df, tanzania_admin1_mpi2_gdf, tanzania_admin2_mpi2_gdf = calculate_mpi_2('Tanzania', admin1_geo_gdf, 'REGION', 'mpi2_admin1', 
        admin2_geo=admin2_geo_gdf, admin2_col='ADM1NAME', admin2_mpi2_col='mpi2_admin2')


# In[ ]:


#preprocess_intermed_data('tanzania', '../input/financial-inclusion-insights/FII_2016_Tanzania_Wave_4_Data.csv')


# In[ ]:


tanzania_fin_df = pd.read_csv('../input/tanzania-preprocessed/tanzania_fii_preprocessed.csv')

# country-specific string processing
tanzania_fin_df['Stratum'] = tanzania_fin_df.Stratum.str.split('_').str[0]
tanzania_fin_df['Stratum'] = tanzania_fin_df['Stratum'].str.title()

# TODO: Update Region strings to match ADM1NAME
#kenya_fin_df['Region'].replace('Muranga', "Murang'a", inplace=True)
#kenya_fin_df['Region'].replace('Tharaka', "Tharaka-Nithi", inplace=True)

tanzania_fin_df.sample(5)


# In[ ]:


tanzania_mpi_fin_df = calculate_mpi2_improved_fin_dim(tanzania_admin2_mpi_df, tanzania_fin_df, 'ADM1NAME', 'Stratum', 'mpi_admin2')


# In[ ]:


# Drop the 2 districts with null fin_poverty and mpi2 (TODO: Investigate why they were null)
tanzania_mpi_fin_df.dropna(axis=0, how='any', inplace=True)


# In[ ]:


check_correlation(tanzania_mpi_fin_df, 'mpi_admin2', 'fin_poverty', 'mpi2')


# A similar result is observed with Tanzania. The financial poverty dimension itself has little correlation with the calculated MPI, however when added (with 1/4 weighting) to the MPI, the correlation between the resulting MPI2 and MPI is very high.

# In[ ]:


plt.subplot(221).set_title("Tanzania County MPI distribuion")
sns.distplot(tanzania_mpi_fin_df.mpi_admin2, bins=30)

plt.subplot(222).set_title("Tanzania County fin_poverty distribuion")
sns.distplot(tanzania_mpi_fin_df.fin_poverty, bins=30)

plt.subplot(223).set_title("Tanzania County MPI2 distribuion")
sns.distplot(tanzania_mpi_fin_df.mpi2, bins=30)

plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=2.0, rect=[0, 0, 0.95, 0.95])


# In[ ]:


tanzania_mpi_fin_df[['mpi_admin2', 'mpi2']].describe()


# The above histograms and summary statistics for MPI and MPI2 show that MPI has a slightly higher standard deviation than MPI2. Both distributions have a similar range of values but the MPI2 is slightly higher than MPI. 

# ### Financial Inclusion Insights Surveys (2014 Surveys, available from WorldBank)
# ***
# In attempting to find other countries to test the new financial dimension calculation on, it was discovered that some of the FII 2016 surveys are missing location information. A query has been sent to Intermedia regarding this. In the meanwhile the author also located survey results from Intermedia available from the World Bank. These are, however, from 2014 and have slightly different features. They are described below, taken from the accompanying documentation.
# 
#     Relevant features:
# 
#         - AA1: Zone	
#         - AA2: District	
#         - AA3: County	
#         - AA4: Sub-county	
#         - AA5: Parish
# 
#         - DL26_1: How many total members does your household have, including adults and children
# 
#         - FL13.1: Do you save with any of the following? Bank 
#         - FL13.2: Do you save with any of the following? Microfinance institution 
#         - FL13.3: Do you save with any of the following? Mobile money 
# 
#         - FL10.1: Do you borrow money from any of the following? Bank
#         - FL10.2: Do you borrow money from any of the following? Microfinance institution
#         - FL10.3: Do you borrow money from any of the following? Mobile money 
#         - FL10.4: Do you borrow money from any of the following? Other formal financial inst
# 
#         - ur: Urban or Rural
#         - own_phone: Do you personally own a mobile phone?
#         - own_SIM: Do you personally own a SIM?
#         - registered_MM: Adults hold a m-money account
#         - registered_bank_full: have a registered full service bank account
#         - registered_NBFI: have a registered full service NBFI account
#        
#     *For all questions, 1=Yes, 2=No

# ### Rwanda<a class="anchor" id="rwanda_mpi2_improved"/>
# ***
# The functions developed for calculation of the improved financial deprivation dimension have been ammended to cope with the 2014 survey format, using Rwanda data initially.

# In[ ]:


def preprocess_wb_intermed_data(country, filepath):
    df = pd.read_csv(filepath)
    df.columns = map(str.lower, df.columns) # Convert column headers to lowercase for consistency
    df = df[['aa1', 'aa2', 'aa3', 'ur', 'dl26_1', 
             'own_phone', 'own_sim', 
             'fl13_1','fl13_2','fl13_3',
             'fl10_1','fl10_2','fl10_3','fl10_4',
             'registered_mm', 'registered_bank_full', 'registered_nbfi']]

    df.rename(columns={'aa1':'zone'}, inplace=True)
    df.rename(columns={'aa2':'district'}, inplace=True)
    df.rename(columns={'aa3':'county'}, inplace=True)
    df.rename(columns={'dl26_1':'total_household_members'}, inplace=True)
    
    df.rename(columns={'fl13_1':'bank_savings'}, inplace=True)
    df.rename(columns={'fl13_2':'mm_savings'}, inplace=True)
    df.rename(columns={'fl13_3':'other_reg_savings'}, inplace=True)

    df.rename(columns={'fl10_1':'bank_borrowing'}, inplace=True)
    df.rename(columns={'fl10_2':'micro_borrowing'}, inplace=True)
    df.rename(columns={'fl10_3':'mm_borrowing'}, inplace=True)
    df.rename(columns={'fl10_4':'other_formal_borrowing'}, inplace=True)
                
    df.to_csv(country+'_fii_preprocessed.csv', index = False)

def process_wb_fii_data(df):
    df['acct_depriv'] = np.where((df['registered_bank_full']==1) | (df['registered_mm']==1), 0, 1)
    df['saving_depriv'] = df[['bank_savings','mm_savings','other_reg_savings']].min(axis=1)
    df['saving_depriv'].replace(1, 0, inplace=True)
    df['saving_depriv'].replace(2, 1, inplace=True)
    
    df['bank_borrowing'].fillna(2, inplace=True)
    df['micro_borrowing'].fillna(2, inplace=True)
    df['mm_borrowing'].fillna(2, inplace=True)
    df['other_formal_borrowing'].fillna(2, inplace=True)
    
    df['borrowing_depriv'] = df[['bank_borrowing','micro_borrowing','mm_borrowing', 'other_formal_borrowing']].min(axis=1)
    df['borrowing_depriv'].replace(1, 0, inplace=True)
    df['borrowing_depriv'].replace(2, 1, inplace=True)
    # Calculate financial deprivation indicator
    # Attempting to keep the definition uniform, lets say that someone is financially deprived if they are deprived in more than one
    # financial indicator.
    df['financial_depriv'] = np.where(df['acct_depriv'] + df['saving_depriv'] + df['borrowing_depriv'] > 1, 1, 0)
    return df

def calculate_mpi2_improved_fin_dim(mpi_df, fin_df, mpi_region_col, fin_region_col, mpi_col):
    fin_df = process_wb_fii_data(fin_df)
    fin_df = calculate_fin_deprivations(fin_df)
    fin_summary_df = aggregate_admin_level(fin_df, level=fin_region_col, col='fin_poverty')
    print("Dataset mpi_df: ", mpi_df.shape)
    mpi_fin_df = mpi_df.merge(fin_summary_df[[fin_region_col, 'fin_poverty']], how='left', left_on=[mpi_region_col], right_on=[fin_region_col])
    print("Dataset mpi_df after merge with fin_df: ", mpi_fin_df.shape)
    mpi_fin_df = calculate_mpi_with_fin_dimension(mpi_fin_df, mpi_col, 'fin_poverty')
    return mpi_fin_df

def check_correlation(mpi_fin_df, mpi_col, fin_poverty_col, mpi2_col):
    # Check Correlation at region level
    print("MPI vs Fin Poverty correlation, p-value: ", pearsonr(mpi_fin_df.loc[:, fin_poverty_col], mpi_fin_df.loc[:, mpi_col]))
    sns.regplot(x=mpi_col, y=fin_poverty_col, data=mpi_fin_df)
    plt.figure()
    print("MPI vs MPI2 correlation, p-value: ", pearsonr(mpi_fin_df.loc[:, mpi2_col], mpi_fin_df.loc[:, mpi_col]))
    sns.regplot(x=mpi_col, y=mpi2_col, data=mpi_fin_df)
    plt.figure()


# In[ ]:


#preprocess_wb_intermed_data('rwanda', '../input/financial-inclusion-insights/final_cgap_rwanda_im_v2.csv')


# In[ ]:


rwanda_fin_df = pd.read_csv('../input/rwanda-preprocessed/rwanda_fii_preprocessed.csv')


# In[ ]:


# Replace region codes with names
fii_zone_map_df = pd.read_csv('../input/rwanda-preprocessed/rwanda_fii_zone_mappings.csv')
rwanda_fin_df['zone'].replace(dict(fii_zone_map_df.values), inplace=True)

fii_district_map_df = pd.read_csv('../input/rwanda-preprocessed/rwanda_fii_district_mappings.csv')
rwanda_fin_df['district'].replace(dict(fii_district_map_df.values), inplace=True)

fii_county_map_df = pd.read_csv('../input/rwanda-preprocessed/rwanda_fii_county_mappings.csv')
rwanda_fin_df['county'].replace(dict(fii_county_map_df.values), inplace=True)

rwanda_fin_df.sample(5)


# In[ ]:


rwanda_fin_df['zone'].replace('Kigali', 'Kigali City', inplace=True)
rwanda_mpi_fin_df = calculate_mpi2_improved_fin_dim(rwanda_admin1_mpi_df, rwanda_fin_df, 'ADM1NAME', 'zone', 'mpi_admin1')


# In[ ]:


check_correlation(rwanda_mpi_fin_df, 'mpi_admin1', 'fin_poverty', 'mpi2')


# In[ ]:


rwanda_mpi_fin_df = calculate_mpi2_improved_fin_dim(rwanda_admin2_mpi_df, rwanda_fin_df, 'NOMDISTR', 'district', 'mpi_admin2')


# In[ ]:


# Drop the 2 districts with null fin_poverty and mpi2 (TODO: Investigate why they were null)
rwanda_mpi_fin_df.dropna(axis=0, how='any', inplace=True)


# In[ ]:


check_correlation(rwanda_mpi_fin_df, 'mpi_admin2', 'fin_poverty', 'mpi2')


# The results for Rwanda show a low correlation between the financial poverty indicator and the calculated MPI. When combined into MPI2 the correlation is again higher but significantly lower than has previously been seen.

# In[ ]:


plt.subplot(221).set_title("Rwanda County MPI distribuion")
sns.distplot(rwanda_mpi_fin_df.mpi_admin2, bins=30)

plt.subplot(222).set_title("Rwanda County fin_poverty distribuion")
sns.distplot(rwanda_mpi_fin_df.fin_poverty, bins=30)

plt.subplot(223).set_title("Rwanda County MPI2 distribuion")
sns.distplot(rwanda_mpi_fin_df.mpi2, bins=30)

plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=2.0, rect=[0, 0, 0.95, 0.95])


# In[ ]:


rwanda_mpi_fin_df[['mpi_admin2', 'mpi2']].describe()


# The histograms and summary statistics for Rwanda MPI and MPI2 show a similar trend as those of Kenya and Tanzania (calculated with the simple financial dimension). The standard deviation is slightly lower for MPI2 and the ranges are very similar, with MPI2 having, in this case significantly, higher values than MPI.

# ### Ghana <a class="anchor" id="ghana_mpi2_improved"/>

# In[ ]:


# Read in DHS and Geo Data
read_data('ghana', 
          '../input/ghana-preprocessed/ghana_household_dhs.csv',
          '../input/ghana-preprocessed/ghana_household_member_dhs.csv',
          '../input/ghana-preprocessed/ghana_births_dhs.csv',
          '../input/ghana-preprocessed/ghana_dhs_cluster.csv',
          '../input/ghana-preprocessed/GHGE71FL.shp', 
          '../input/ghana-humdata-admin-geo/GHA_admbndp1_1m_GAUL.shp', 
          '../input/ghana-humdata-admin-geo/GHA_admbndp2_1m_GAUL.shp')

# Simplify geometry.     
replace_geometry(admin2_geo_gdf, '../input/ghana-humdata-admin-geo/GHA_admbndp2_1m_GAUL_simple.shp')

ghana_admin1_mpi_df, ghana_admin2_mpi_df, ghana_admin3_mpi_df = calculate_mpi('Ghana', admin1_geo_gdf, 'ADM1_NAME', 'mpi_admin1', admin2_geo=admin2_geo_gdf, admin2_col='ADM2_NAME', admin2_mpi_col='mpi_admin2')


# In[ ]:


#preprocess_wb_intermed_data('ghana', '../input/financial-inclusion-insights/final_cgap_ghana_im_v2.csv')


# In[ ]:


ghana_fin_df = pd.read_csv('../input/ghana-preprocessed/ghana_fii_preprocessed.csv')


# In[ ]:


# Replace region codes with names
fii_zone_map_df = pd.read_csv('../input/ghana-preprocessed/ghana_fii_region_mappings.csv')
ghana_fin_df['zone'].replace(dict(fii_zone_map_df.values), inplace=True)


# In[ ]:


ghana_mpi_fin_df = calculate_mpi2_improved_fin_dim(ghana_admin1_mpi_df, ghana_fin_df, 'ADM1_NAME', 'zone', 'mpi_admin1')


# In[ ]:


check_correlation(ghana_mpi_fin_df, 'mpi_admin1', 'fin_poverty', 'mpi2')


# Unfortunately, there were many mismatches between the administrative level 2 region names in the two datasets (DHS and FII) so the calculation and analysis has been run at administrative level 1 only. This can easily be remedied by searching for the correct mappings and updating the FII data, however this was not done due to time constraints.
# 
# Nevertheless, the comparison at the administrative level 1 for Ghana shows similar results to those of Rwanda. There is moderate correlation between the financial poverty measure and the calculated MPI but when the MPI is combined with the financial deprivation dimension, there is a high correlation between MPI2 and MPI.

# In[ ]:


plt.subplot(221).set_title("Ghana Region MPI distribuion")
sns.distplot(ghana_mpi_fin_df.mpi_admin1, bins=30)

plt.subplot(222).set_title("Ghana Region fin_poverty distribuion")
sns.distplot(ghana_mpi_fin_df.fin_poverty, bins=30)

plt.subplot(223).set_title("Ghana Region MPI2 distribuion")
sns.distplot(ghana_mpi_fin_df.mpi2, bins=30)

plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=2.0, rect=[0, 0, 0.95, 0.95])


# In[ ]:


ghana_mpi_fin_df[['mpi_admin1', 'mpi2']].describe()


# Again, similar results are observed when examining Ghana MPI and MPI2 histograms and summary statistics (even though administrative level 1 is being used here instead of level 2). 

# ## 6. Conclusion <a class="anchor" id="conclusion"/>
# ***
# It is possible to develop arbitrarily many different indices to describe poverty but how can it be determined whether one index is better than another? I had hoped, with this notebook, to demonstrate that the developed MPI2 has a significantly larger standard deviation or range compared to the MPI, in order to argue that it is better able to differentiate between poverty in various regions. This is unfortunately not conclusively the case. 
# 
# One consistent factor among all the countries studied in this notebook is that including the financial dimension increases the poverty score of most regions. That means, it increases either or both of:
#     1.     the headcount ratio of people considered to be multidimensionally poor
#     2.     the total poverty intensity among those considered to be multidimensionally poor.
# This makes sense, as adding another dimension to the index includes another group of people as muiltidimensionally poor, ie: those who previously were only poor in one of the indicators (health, education or living standard) and are also considered financially deprived. And for those already previously considered multidimensionally poor, it would increase their poverty intensity if they are also financially deprived.
# 
# It would have been good to also have some test data (country) where the opposite is true and considering financial deprivation decreases the overall poverty score, for comparison. However, such a country has not been found during the course of this project due to limited resources (time). There are other datasets available that could have been processed to perhaps look at countries with quite low MPIs or the highest MPIs. This would however, have required changes to the preprocessing functions that work quite generically for the countries that have been looked at (due to all of them having data for the same DHS survey version and FII survey data). But with more time available, the functions can certainly be generalised to be able to process different datasets and analyse the score distributions for more countries.
# 
# Despite lack of statistical evidence to support use of the MPI2, I would still argue that it is a better index than MPI for Kiva's purposes. By looking at the characteristics of those who a given poverty measure would include, or would leave out, we can provide direct evidence on whether that measure does a better job of capturing the disadvantaged. The MPI2 includes those who are multidimensionally deprived, where financial deprivation counts as a dimension, and this being the area where Kiva can directly make an impact by providing financial inclusion, should count for a lot.
# 
# Should Kiva wish to use this new index, the functions developed in this notebook (and Part III notebook) should work for all countries that have DHS survey v7 data and Intermedia Financial Inclusion Insights survey data available. Additionally, the code is written in a modular way so that only minor modifications should be required, should there be other datasets available providing similar features.

# ## 7. References <a class="anchor" id="references"/>
# ***
# - [Demographic and Health Surveys Program Website](https://dhsprogram.com/Data/)
# - [Demographic and Health Surveys Program Recode (PDF)](https://dhsprogram.com/pubs/pdf/DHSG4/Recode6_DHS_22March2013_DHSG4.pdf)
# - [HDR Technical Notes (PDF)](http://dev-hdr.pantheonsite.io/sites/default/files/hdr2016_technical_notes_0.pdf)
# - [Financial Inclusion Program](http://finclusion.org/)
# - [Ghana - Financial Inclusion Insights Survey 2014](http://microdata.worldbank.org/index.php/catalog/2730)
# - [Intermedia Website](http://www.intermedia.org/)
# - [Kiva Website](https://www.kiva.org/)
# - [Mapshaper](http://mapshaper.org/)
# - [Rwanda - Financial Inclusion Insights Survey 2014](http://microdata.worldbank.org/index.php/catalog/2726/study-description)
# - [The Humanitarian Data Exchange](https://data.humdata.org/)
# - [UNDP Specifications for Computation of the MPI (PDF)](http://hdr.undp.org/sites/default/files/specifications_for_computation_of_the_mpi_0.pdf)
