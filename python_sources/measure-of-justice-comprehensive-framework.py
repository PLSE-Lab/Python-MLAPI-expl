#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').show();
 } else {
 $('div.input').hide();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')


# # Problem statement.
# 
# Kaggle competition @
# https://www.kaggle.com/center-for-policing-equity/data-science-for-good
# 
# *How do you measure justice? And how do you solve the problem of racism in policing? We look for factors that drive racial disparities in policing by analyzing census and police department deployment data. The ultimate goal is to inform police agencies where they can make improvements by identifying deployment areas where racial disparities exist and are not explainable by crime rates and poverty levels.*
# 

# ### Exploring Data
# Dataset consists of number of folders one for a department. Department is a number in XX-YYYYY format. Each folder has     
# 1. XX-YYYYY_ACS_data -> ACS socio-economic data.   
# 2. XX-YYYYY_Shapefiles -> Contains group of files needed for Shapefiles of police departments containing geographic information. .shp has geometries.  
# 3. XX-YYYYY_(*variable name*).csv -> containing police records.  
# 
# Quick scan of files and as also specified in the problem statement, data is not consistent/uniform accross folders. Also geospatial data(coordinates) does not seem to follow same standards even within a department data.  
# 
# **ACS_data ->** There are 7 folders named in  XX-YYYYY_ACS_*(variable_name)* format. Folders are for seperate ACS socio-economic survey such as education attainment, povert, income, housing etc. Each of the folders have 2 files. 1 for metadata and other for actual survey data.  
#     
#     **Metadata ->** Lists codes of the column names and meaning of column codes. Quick scan of data shows, each of the data set has anywhere around 700 columns. They are different for different ACS surveys. *This indicates need for automated processeing and identifying of data*.  
#     
#     **Data ->** These files have censsus track number and survey data as indicated in corresponding metadata sheet. Number of columns is high in upto 800 as well and differ from one ACS to other. There is some pattern though like, it has estimate number, percent, margin of error for each. Also there is different slice and dice of survey data by different socio economic characteristics such as gender, race age etc. This can be exploited in automated processing of metadata. 
# 
# **UOF ->** Police records have records of police incidents, vehicle stops in the region. The format changes widely accross police departments. Usually the records have gender, sex of the subject, nature of incident, location of the incident and in some cases police officers involved.  
# 
# **Shapefiles ->** Associated with UOF file is the Shapefiles which contain geospatial data of the corresponding police department.

# ### Solution framework.  
# One of the key feature of the solution should be repeatability accross various departments data amid varying data formats.  
# We need to have reusable functions to bring together all data into common format which can be run through analysis and get insights.  
# Below are the lists of common modules/steps created to facilitate analysis repeatability accross departments.  
# 
# **1. step1_police_recs.**  
# 
#     As the shapefiles and police records vary a lot in format/column names accross departments. We need to join these 2 dataframes to begin geospatial analsysis. So the utitlity will analyse the columns and gives its suggestion for columns which are potential common keys i.e. they can be used for joining the datasets.  
#     
#     Input -  
#         path of the department folder.  
#     Output -  
#         pandas dataframe for police records.  
#         geopandas dataframe with shapefiles. 
#         Suggestion of columns for common keys.
#         
# 
# **2. step2_crime_summary**.  
#     
#     Merge shapefiles and police records summarised. 
#     
#     Input - 
#         pr_df - police record dataframe
#         pr_cols_summary - police record columns to be used for summarization.
#         pr_common_key - common Key from police records for joining,
#         pg_df - police shapefile dataframe.
#         pg_common_key - common key from police shapefile geodataframe for joining.
#     
#     Output -  
#         geopandas dataframe with CRS set to EPSG:32118. police records summarised by cols_summary passed.  
#         
#         EPSG:32118 has base units in meters. It is chooses as we need to calculate area and get % overlap. Area in meter squared are more comprehensible and this is bigger numeric number than area in degrees (when EPSG is choosen with degress as base units). Below website has good informa
#           
# **3. step3_visualize_police_records**.  
# 
#     Choropleth plot of police records by precinct/district.
#     Bubble size is the record totals. Each value of summary column
#     
#     Input -  
#         geodataframe of police records. (step 2 output)
#     
#     Output -  
#         seriese of choropleth plots.
# 
# **4. step4_merge_census_crimes**.  
#     census.gov website has geospatial data for each of the states. https://www.census.gov/geo/maps-data/data/cbf/cbf_tracts.html  
#     Download census track from external source. ACS survey data is by census track. So we need to merge police precinct and ACS track. Police precinct and ACS survey overlap and there is many to many relationship. Different % of ACS tracks overlap with police precincts. This step will also return data frame of police precincts and overlapping census tracks with % overlap.  
#     
#     Input - 
#         Path of census tracks file (.shp)
#         Police precinct data frame rolledup and the common key most likely Dist/precinct key.
# 
#     Output - 
#         Map of state census track overlapped with police precinct.
#         Map of only overlapping census tracks and police precincts.
#         data frame of police precinct common key, census track key and % overlap of census track on police precinct.
#     
# **5. step5_acs_data_summary**.  
#  
#  There are around 700 columns in each of the 7 folders. Studied all the columns via utility which slices and dices codes to have better clarity on the content of ACS data. After analysis below are the columns extracted for analsysis.
#  Utility function used is 'utility_metadata_summary'
#  *Education attainment.*   
#  
#     1. HC01_EST_VC02 + HC01_EST_VC08 -> Total population above 18 years.
#     2. HC01_EST_VC03 + HC01_EST_VC09 + HC01_EST_VC10 -> Up to highschool educated.
# 
#  *Education Attainment above 25.*  
#  
#     EA has data for population above 18 which includes above 25 population as well. So this dataset is redundant. Not including in the datasets.  
# 
#  *Employment.*  
#  
#     1. HC01_EST_VC01 -> Total population above age 16.  
#     2. HC03_EST_VC01 -> Employment to population ratio. Calculate Employed population as we need to rollup to police precinct level.
# 
#  *Income.*  
#  
#     Data set has median income values. Couldn't find statistically meanigful way yo rollup/take proportion of median income from ACS to police precinct levels. Ignoring this dataset.  
# 
#  *Owner occupied housing.*  
#  
#     1. HC03_EST_VC01 -> renter occupiedhousing units estimate.  
#     2. HC02_EST_VC01 -> Owner occupied housing unit.
# 
# *Poverty.*  
#     
#     1.HC01_EST_VC01 -> Total population for measuring poverty.  
#     2.HC02_EST_VC01 -> population below poverty level.
# 
#  *Race, sex and Age.*  
#     1. HC01_VC03 -> Total population.  
#     2. HC01_VC04 -> Male population.  
#     3. HC01_VC05 -> Female population.  
#     4. HC01_VC08 + HC01_VC09 + HC01_VC10 -> upto Age 14.  
#     5. HC01_VC29 -> Age 65 and over.  
#     6. Calculate 15-64 age group.  
#     7. HC01_VC44 -> one race population.  
#     8. HC01_VC49 -> white_race.  
#     9. HC01_VC50 -> black race.  
#     10. HC01_VC51 -> american indian race.  
#     11. HC01_VC56 -> asian.  
#     12 .HC01_VC88 -> latino.
# 
# **6. step6_merge_acs_pp_data**  
# Merge dervied ACS dataframe with summarised police records to get patterns.  
# 
# **7. step7_append_geo**  
# Append geometry and geospatial information.
# 
# **8. ste8_corr_plot**  
# Get correlation of derived matrices and police records. IS there any relationship?  
# 
# **9. step9_elbow_plot**  
# Run a clustering algorith to see if police districts cluster together based on similarities on derived ACS and police records data. This will help better udnerstand any displarities. 
# 
# **10. step10_cluster_and_corr**  
# Get correlation of derived matrices and police records. IS there any relationship?
# 
# **11. Bias**  Get % of police records and % of population for that group under study. This may help understand if there is racial bias or gender bias etc.  
# 
# **12. True crimes?**  Police records contain many features which are used to get true crimes i.e. incidents caused by a report, reasonable suspicion etc vs 'investigative' records which are preventive. Get a stat of percent of total police records which are preventive type vs true crime. Displarity in this metric may indicate over policing.

# **Libraries loaded are**  
# 1. Geopandas and shapely for geospatial analysis.  
# 2. numpy, pandas for data wrangling.  
# 3. matplotlib and seaborn for plotting.  
# 4. sklearn for machine learning.  
# 5. glob and re for miscellaneous.

# In[ ]:


#Load libraries needed for the solution
get_ipython().run_line_magic('matplotlib', 'inline')
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


# In[ ]:


def step1_police_recs(folder_name):
    ''' This function reads police records and police distrinct/precincts shapefiles.
    Returns pandas dataframe for police records and geopandas dataframe for shapefiles.
    Message for common key column between these 2 data sets is returned. 
    Method : Check for unique values in each column of both datasets. Return columns from datasets where unique values are same.
    If none are found next best matches are returned with rate of match if any.
    Input : folder path of the department.
    Output : 2 data sets. 
        First pandas dataframe for police records.
        Second geopandas dataframe for police precincts.
        Message for common key.
        Message for CRS of geopandas.
    '''
    file = glob.glob(folder_name+'*.csv')
    if(len(file)!=1):
        raise ValueError('There should be only one file with .csv extension')
    police_records = pd.read_csv(file[0],skiprows = 1)
    police_records = police_records.apply(pd.to_numeric, errors = 'ignore')
    print('\npolice_records\n')
    print(police_records.head(3))
    file = glob.glob(folder_name+'*Shapefiles/*.shp')
    police_geo = gpd.read_file(file[0])
    c1 = police_geo.crs
    police_geo = police_geo.apply(pd.to_numeric, errors='ignore')
    police_geo.crs = c1

    print('\npolice_shapefiles\n')
    print(police_geo.head(3))

    if(len(police_geo.crs)==0):
        print(f'\npolice_geo. CRS is not set. Please set it manually by inspecting values.\n')
    else: 
        print(f'\npolice_geo. CRS is {police_geo.crs}\n')
    flg = 0
    pgcol_dicts = {}
    for c in police_geo.columns:
        if(c != 'geometry'):
            pgcol_dicts[c] = police_geo[c].unique()

    prcol_dicts = {}
    for c in police_records.columns:
        if(c != 'geometry'):
            prcol_dicts[c] = police_records[c].unique()

    for k1 in pgcol_dicts:
        for k2 in prcol_dicts:
            if(set(prcol_dicts[k2]) == set(pgcol_dicts[k1])):
                print('\nCommon key found are\n')
                print(f'\n\'{k1}\' and \'{k2}\'\n')
                flg = 1
    if(flg == 0):
        print('\n\nPerfect common key is not found.')
        for k1 in pgcol_dicts:
            for k2 in prcol_dicts:
                a = set(prcol_dicts[k2]) 
                b = set(pgcol_dicts[k1])
                common = a.intersection(b)
                rate = np.round((len(common)/len(a))*100)
                if(rate > 0):
                    print(f'\n     Suggestion -- \'{k1}\' and \'{k2}\' with common rate of {rate}%\n')
                    print(f'     Common values are {common}')
    return(police_geo,police_records)


# In[ ]:


def step2_crime_summary(pr_df, pr_cols_summary,pr_common_key, pg_df, pg_common_key):
    '''Rollup police records by common key usually police district/police precinct
    Columns to be summarised from police records are passed. % of each value of the column are calculated at 
    common key level. Join with police shapefiles. Change crs to EPSG:32118.
    Input - pr_df - police record dataframe
    pr_cols_summary - police record columns to be used for summarization.
    pr_common_key - common Key from police records,
    pg_df - police shapefile dataframe.
    pr_common_key - common key from police shapefile geodataframe.
    '''
    t = pd.DataFrame(pr_df[pr_common_key].value_counts().reset_index())
    t.columns = [pr_common_key,'total_police_records']
    for c in pr_cols_summary:
        t1=pr_df[[pr_common_key,c]].groupby(by=[pr_common_key,c]).size().reset_index()
        t1 = t1.pivot(index = pr_common_key,columns = c,values = 0).reset_index()
        t = t.merge(t1,on = pr_common_key,how = 'left')
    pg = pd.merge(pg_df[[pg_common_key,'geometry']], t, left_on = pg_common_key, right_on = pr_common_key)
    #pg.drop(columns = [pr_common_key],inplace = True)
    pg = gpd.GeoDataFrame(pg,
                         geometry = 'geometry',
                         crs = pg_df.crs)
    pg.to_crs({'init': 'epsg:32118'},inplace = True)
    pg.fillna(value=0,inplace = True)
    return(pg)


# In[ ]:


def step4_merge_census_crimes(census_path, crimes, crimes_key,list_id = None, tpath = None):
    '''Input - Path of census tracks file (.shp)
    Police precinct data frame rolledup and the common key most likely Dist/precinct key.
    Output - Map of state census track overlapped with police precinct.
    Map of only overlapping census tracks and police precincts.
    Returns - data frame of police precinct common key, census track key and % overlap of census track
    on police precinct.
    '''
    cf_path = glob.glob(census_path +'/*.shp')
    if(len(cf_path) != 1) :
        raise ValueError('There should be one and only one .shp file in the folder')
    census = gpd.read_file(cf_path[0])
    census.to_crs(crs = {'init': 'epsg:32118'}, inplace = True )
    
    #Plot overlapping map plot.
    #fig,(ax0,ax1) = plt.subplots(nrows = 1,ncols = 2,figsize = (10,5))
    ax0 = census.plot(figsize = (12,12), facecolor = 'None', edgecolor = 'grey')
    crimes.plot(ax=ax0, edgecolor = 'black')
    ax0.set_axis_off()
    ax0.set_title('State census map overlapped with police precinct map',fontsize =14)
    #Calculate area
    census['area_acs'] = census['geometry'].area
    census = census[['AFFGEOID','geometry','area_acs']]
    if(list_id):
        census = census[census.AFFGEOID.isin(list_id)]

 # Geopandas overlay/merge still not working on kaggle. A localcopy after exact below operation is uploaded and given path here.   
#    t = gpd.overlay(crimes[[crimes_key,'geometry']], 
#                census[['AFFGEOID','geometry','area_acs']], 
#                how = 'intersection')
    t = gpd.read_file(tpath)
    t['overlap_area'] = t['geometry'].area
    t['percent_share'] = t['overlap_area']/t['area_acs'] *100
    t = t.filter(items = [crimes_key,'AFFGEOID','percent_share']).sort_values(by = [crimes_key,'AFFGEOID']).reset_index(drop=True)
    
    t1 = pd.merge(t, census,on = 'AFFGEOID',how = 'left')
    ax1 = t1.plot(figsize = (12,12), facecolor = 'None', edgecolor = 'grey')
    crimes_by_precinct.plot(ax=ax1, column = crimes_key, alpha = 0.5, edgecolor = 'black',legend = True, cmap = 'terrain')
    ax1.set_title('ACS Census track overlayed with police precincts ', fontsize = 14)
    ax1.set_axis_off()
    ax1.get_legend().set_title('Police distrincts')
    ax1.get_legend().set_bbox_to_anchor((1.4,1))

    for i, pp in enumerate(crimes_by_precinct[crimes_key]):
        ax1.annotate(pp,
                    (crimes_by_precinct.geometry[i].centroid.x, crimes_by_precinct.geometry[i].centroid.y),
                   fontsize =14, color = 'r', weight = 'bold')
    plt.tight_layout()
    return(t)    


# In[ ]:


def step5_acs_data_summary(folder_path):
    f = glob.glob(folder_name + '*ACS_data/*ACS_education-attainment/ACS_15_*_with_ann.csv')
    acs_ea = pd.read_csv(f[0], skiprows = [1])
    acs_ea_total_pop = pd.DataFrame()
    acs_ea_total_pop['GEO.id'] =  acs_ea['GEO.id']
    #acs_ea_total_pop['ea_pop_above_18'] =  acs_ea['HC01_EST_VC02'] + acs_ea['HC01_EST_VC08']
    acs_ea_total_pop['ea_pop_upto_highschool'] =  acs_ea['HC01_EST_VC03'] + acs_ea['HC01_EST_VC09'] + acs_ea['HC01_EST_VC10']

    f = glob.glob(folder_name + '*ACS_data/*ACS_employment/ACS_15_*_with_ann.csv')
    acs_emp = pd.read_csv(f[0], skiprows = [1],na_values = '-')
    acs_emp_total_pop = pd.DataFrame()
    acs_emp_total_pop['GEO.id'] =  acs_emp['GEO.id']
    #acs_emp_total_pop['emp_pop_above_16'] =  acs_emp['HC01_EST_VC01']
    acs_emp_total_pop['emp_pop_employed'] =  acs_emp['HC01_EST_VC01'] * acs_emp['HC03_EST_VC01']/100

    f = glob.glob(folder_name + '*ACS_data/*ACS_owner-occupied-housing/ACS_15_*_with_ann.csv')
    acs_ooh = pd.read_csv(f[0], skiprows = [1],na_values = '-')
    acs_ooh_total_pop = pd.DataFrame()
    acs_ooh_total_pop['GEO.id'] =  acs_ooh['GEO.id']
    acs_ooh_total_pop['ooh_owner_occupied_units'] =  acs_ooh['HC02_EST_VC01']
    acs_ooh_total_pop['ooh_renter_occupied_units'] =  acs_ooh['HC03_EST_VC01']

    f = glob.glob(folder_name + '*ACS_data/*ACS_poverty/ACS_15_*_with_ann.csv')
    acs_pov = pd.read_csv(f[0], skiprows = [1],na_values = '-')
    acs_pov_total_pop = pd.DataFrame()
    acs_pov_total_pop['GEO.id'] =  acs_pov['GEO.id']
    acs_pov_total_pop['pov_total_pop'] =  acs_pov['HC01_EST_VC01']
    acs_pov_total_pop['pov_below_poverty'] =  acs_pov['HC02_EST_VC01']

    f = glob.glob(folder_name + '*ACS_data/*ACS_race-sex-age/ACS_15_*_with_ann.csv')
    acs_rsa = pd.read_csv(f[0], skiprows = [1],na_values = '-')
    acs_rsa_total_pop = pd.DataFrame()
    acs_rsa_total_pop['GEO.id'] =  acs_rsa['GEO.id']
    acs_rsa_total_pop['rsa_total_pop'] =  acs_rsa['HC01_VC03']
    acs_rsa_total_pop['rsa_male_pop'] =  acs_rsa['HC01_VC04']
    acs_rsa_total_pop['rsa_female_pop'] =  acs_rsa['HC01_VC05']
    acs_rsa_total_pop['rsa_age_upto14_pop'] =  acs_rsa['HC01_VC08'] + acs_rsa['HC01_VC09'] + acs_rsa['HC01_VC10']
    acs_rsa_total_pop['rsa_age_above_65_pop'] =  acs_rsa['HC01_VC29']
    acs_rsa_total_pop['rsa_age_15_64_pop'] =  acs_rsa_total_pop['rsa_total_pop'] - acs_rsa_total_pop['rsa_age_upto14_pop'] - acs_rsa_total_pop['rsa_age_above_65_pop']
    #acs_rsa_total_pop['rsa_one_race_pop'] =  acs_rsa['HC01_VC44']
    acs_rsa_total_pop['rsa_white_race_pop'] =  acs_rsa['HC01_VC49']
    acs_rsa_total_pop['rsa_black_race_pop'] =  acs_rsa['HC01_VC50']
    acs_rsa_total_pop['rsa_ameindia_race_pop'] =  acs_rsa['HC01_VC51']
    acs_rsa_total_pop['rsa_asian_race_pop'] =  acs_rsa['HC01_VC56']
    acs_rsa_total_pop['rsa_latino_race_pop'] =  acs_rsa['HC01_VC88']

    acs_total_pop = pd.merge(acs_rsa_total_pop, 
             acs_pov_total_pop, 
             on = 'GEO.id', 
             how = 'left').merge(acs_ooh_total_pop, 
                                 on = 'GEO.id', 
                                 how = 'left').merge(acs_emp_total_pop,
                                                    on = 'GEO.id',
                                                    how = 'left').merge(acs_ea_total_pop,
                                                                       on = 'GEO.id',
                                                                       how = 'left')

    acs_total_pop['pov_below_poverty_calc'] = acs_total_pop['pov_below_poverty']/acs_total_pop['pov_total_pop']*acs_total_pop['rsa_total_pop']
    acs_total_pop.drop(columns = ['pov_below_poverty','pov_total_pop'],inplace = True)
    return(acs_total_pop)


# In[ ]:


def step6_merge_acs_pp(pp_census_overlap, acs_total_pop,common_key = 'DISTRICT'):
    pp_acs_total_pop = pp_census_overlap.merge(acs_total_pop,left_on = 'AFFGEOID', right_on = 'GEO.id', how = 'inner')
    for c in pp_acs_total_pop.iloc[:,4:].columns:
        pp_acs_total_pop[c] = pp_acs_total_pop[c] * pp_acs_total_pop['percent_share']

    pp_acs_total_pop = pp_acs_total_pop.drop(columns = ['AFFGEOID','percent_share','GEO.id']).groupby(by = common_key).sum().reset_index()
    print(pp_acs_total_pop.head())
    return(pp_acs_total_pop)


# In[ ]:


def step7_append_geo(crimes_by_precinct,pp_acs_total_pop, common_key = 'DISTRICT', size_divider = 10):
    crimes_pp_total_pop = crimes_by_precinct[[common_key,
                                              'geometry',
                                              'total_police_records']].merge(pp_acs_total_pop, 
                                                                             on = common_key,
                                                                             how = 'left')
    col_series = crimes_pp_total_pop.iloc[:,3:].columns
    choropleth_map(crimes_pp_total_pop,
                   col_series,
                   bubble = 'total_police_records',
                   lab_title='Population range',
                  size_divider = size_divider)
    return(crimes_pp_total_pop)


# In[ ]:


def ste8_corr_plot(crimes_pp_total_pop):
    fig, ax = plt.subplots(figsize = (12,12))
    sns.heatmap(crimes_pp_total_pop.corr(),vmin=-1,vmax=1,annot = True,cmap = 'Blues',ax = ax)
    plt.title('Correlation chart of ACS data and total police records', fontsize=16)
    return


# In[ ]:


def step9_elbow_plot(crimes_pp_total_pop,max_size = 10):
    t22 = crimes_pp_total_pop.drop(columns = ['geometry'])
    scale = MinMaxScaler()
    t22_scaled = t22.iloc[:,2:]
    t22_scaled = scale.fit_transform(t22.iloc[:,2:])
    ranges = np.arange(2,max_size)
    inert = []
    for k in ranges:
        model = KMeans(n_clusters = k)
        model.fit(t22_scaled)
        inert.append(model.inertia_)

    plt.plot(ranges,inert,'-o')
    plt.xticks(ranges)
    plt.title('Elbow plot', fontsize = 16, loc = 'center')
    plt.show()
    return(t22_scaled)


# In[ ]:


def step10_cluster_and_corr(crimes_pp_total_pop, num_clust = 2,common_key = 'DISTRICT'):
    t22 = crimes_pp_total_pop.drop(columns = ['geometry'])
    scale = MinMaxScaler()
    t22_scaled = t22.iloc[:,2:]
    t22_scaled = scale.fit_transform(t22.iloc[:,2:])
    good_model = KMeans(n_clusters = num_clust)
    good_model.fit(t22_scaled)
    t22['cluster_scaled'] = good_model.predict(t22_scaled)
    pca = PCA(n_components = 2)
    t = pca.fit_transform(t22_scaled)
    pca_df = pd.DataFrame(data = t, columns = ['PCA1','PCA2'])
    pca_df = pd.concat([pca_df,t22['cluster_scaled']],axis=1)
    fig, (ax0,ax1) = plt.subplots(nrows=1,ncols=2,figsize= (16,8))
    ax0.scatter(pca_df.PCA1,pca_df.PCA2,c = pca_df.cluster_scaled,s=60, alpha = 0.7, cmap = 'tab20')
    ax0.set_title('Clustering of the police precincts', fontsize = 16)
    ax0.set_xlabel('PCA 1', fontsize =16)
    ax0.set_ylabel('PCA 2', fontsize =16)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    for i, pp in enumerate(t22[common_key]):
        ax0.annotate(pp,(pca_df.PCA1[i]+0.01,pca_df.PCA2[i]-0.01),fontsize=14)

    temp = crimes_pp_total_pop.merge(t22[[common_key,'cluster_scaled']],on =common_key)
    temp.plot(ax = ax1, column = 'cluster_scaled', cmap = 'tab20', categorical = True,edgecolor='grey',legend = True)
    ax1.get_legend().set_title('Clusters')
    ax1.get_legend().set_bbox_to_anchor((1,1))
    ax1.set_axis_off()
    ax1.set_title('Police districts color coded by groupings based on similarity',fontsize=16)
    for i, pp in enumerate(crimes_pp_total_pop[common_key]):
        ax1.annotate(pp,(crimes_pp_total_pop.geometry[i].centroid.x, crimes_pp_total_pop.geometry[i].centroid.y))

    t= pd.DataFrame(data = temp[temp.cluster_scaled == 0].corr().iloc[0])
    t.rename(columns ={'total_police_records' : 'cluster ' + str(0)},inplace = True)
    for i in np.arange(num_clust-1):
        t = pd.concat([t, temp[temp.cluster_scaled == i+1].corr().iloc[0]],axis=1)
        t.rename(columns ={'total_police_records' : 'cluster ' + str(i+1)},inplace = True)
    print(t)
    return(temp)


# In[ ]:


def step11_bias_calc(crimes_by_precinct, pp_acs_total_pop, common_key = 'DISTRICT'):
    crimes_by_precinct_percent = pd.concat([crimes_by_precinct[common_key],
                                            crimes_by_precinct.MALE/(crimes_by_precinct.MALE+crimes_by_precinct.FEMALE)*100,
                                           crimes_by_precinct.FEMALE/(crimes_by_precinct.MALE+crimes_by_precinct.FEMALE)*100,
                                           crimes_by_precinct.asian/(crimes_by_precinct.total_police_records)*100,
                                           crimes_by_precinct.black/(crimes_by_precinct.total_police_records)*100,
                                           crimes_by_precinct.hispanic/(crimes_by_precinct.total_police_records)*100,
                                           crimes_by_precinct.native_indian/(crimes_by_precinct.total_police_records)*100,
                                           crimes_by_precinct.white/(crimes_by_precinct.total_police_records)*100],
                                           axis = 1)
    crimes_by_precinct_percent.columns = [common_key,
                                          'male_percent',
                                          'female_percent',
                                          'asian_percent',
                                          'black_percent',
                                         'hispanic_percent',
                                         'native_american_percent',
                                         'white_percent']
    pp_acs_total_pop_percent = pd.concat([pp_acs_total_pop[common_key],
                                            pp_acs_total_pop.rsa_male_pop/(pp_acs_total_pop.rsa_total_pop)*100,
                                           pp_acs_total_pop.rsa_female_pop/(pp_acs_total_pop.rsa_total_pop)*100,
                                           pp_acs_total_pop.rsa_asian_race_pop/(pp_acs_total_pop.rsa_total_pop)*100,
                                           pp_acs_total_pop.rsa_black_race_pop/(pp_acs_total_pop.rsa_total_pop)*100,
                                           pp_acs_total_pop.rsa_latino_race_pop/(pp_acs_total_pop.rsa_total_pop)*100,
                                           pp_acs_total_pop.rsa_ameindia_race_pop/(pp_acs_total_pop.rsa_total_pop)*100,
                                           pp_acs_total_pop.rsa_white_race_pop/(pp_acs_total_pop.rsa_total_pop)*100],
                                           axis = 1)
    pp_acs_total_pop_percent.columns = [common_key,
                                          'male_percent',
                                          'female_percent',
                                          'asian_percent',
                                          'black_percent',
                                         'hispanic_percent',
                                         'native_american_percent',
                                         'white_percent']
    for c in pp_acs_total_pop_percent.columns[1:]:
        temp = pp_acs_total_pop_percent[[common_key,c]].merge(crimes_by_precinct_percent[[common_key,c]],
                                                                    on = common_key,
                                                                   suffixes = ('_pop','_polrec'))
        temp['diff'] = temp.iloc[:,1] - temp.iloc[:,2]
        cleveland_plot(temp)
    
    return


# In[ ]:


def choropleth_map(df, c, bubble = None, fsize = (16,40), lab_title = 'range', color_map = 'Blues',size_divider = 10):
    '''Plot series of choropleths in 2 columns settings. 'quantiles' scheme and cmap = 'Blues' are used.
    Input - df - dataframe.
            c - list of columns to be used for choropleth.
            bubble - column which should be used for plotting bubbles.
            fsize = tupple of fig size.        
    '''
    plt_rows = np.ceil(len(c)/2).astype(int)
    fig, axes = plt.subplots(nrows = plt_rows, ncols =2, sharex = True, sharey = True, figsize = fsize)
    plt.Axes
    i = 0
    for row_ax in axes:
        for ax in row_ax:
            ax.set_axis_off()
            if i < len(df.columns)-3:
                df.plot(ax = ax, column = c[i], legend=True,  cmap = color_map, scheme = 'quantiles',edgecolor = 'black')
                ax.set_title('Map - ' +c[i], fontsize=16)
                ax.get_legend().set_bbox_to_anchor((1.4,1))
                ax.get_legend().set_title(lab_title)
                if(bubble):
                    ax.scatter(x=df['geometry'].centroid.apply(lambda x: Point(x).x),
                            y=df['geometry'].centroid.apply(lambda x: Point(x).y),
                            s = (df[bubble]/size_divider),
                            alpha=0.4,
                            c='red')
                i = i+1
    plt.tight_layout()


# In[ ]:


def cleveland_plot(df1, fsize=(10,5)):
    ''' Plot a cleveland plot showing the xmin and xmax for a set of categorical variable'''
    tot_recs = 10 if len(df1) > 10 else len(df1)
    y_range = np.arange(1,tot_recs+1)
    

    df1.sort_values(by = 'diff',inplace = True)
    df = df1.iloc[:tot_recs]
    fig,(ax0,ax1) = plt.subplots(nrows = 1, ncols=2,figsize = fsize)
    ax0.hlines(y = y_range, xmin = df.iloc[:,1], xmax = df.iloc[:,2], 
               alpha=0.3, color='gray', linewidth = 2)
    ax0.scatter(x = df.iloc[:,1],  y = y_range, color = 'green', label = df.columns[1], s =50, marker='o')
    ax0.scatter(x = df.iloc[:,2],  y = y_range, color = 'red', label = df.columns[2], s = 50,marker = 'H')
    ax0.set_yticks(y_range) 
    ax0.set_yticklabels(df1.iloc[:,0])
    ax0.set_ylabel('police districts', fontsize = 14, color = 'gray')
    ax0.set_ylim(tot_recs+1,0)
    ax0.set_xlabel('Percent', fontsize = 14, color = 'gray')
    ax0.legend(fontsize = 12,loc='best')
    ax0.set_title('Police rec % > pop % \nBias (pop - pol_rec) ' + df.columns[1], fontsize = 14)
    plt.grid(False)
    ax0.set_facecolor('white')
    fig.set_facecolor('w')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    
    df1.sort_values(by = 'diff',inplace = True, ascending = False)
    df = df1.iloc[:tot_recs]
    ax1.hlines(y = y_range, xmin = df.iloc[:,1], xmax = df.iloc[:,2], 
               alpha=0.3, color='gray', linewidth = 2)
    ax1.scatter(x = df.iloc[:,1],  y = y_range, color = 'green', label = df.columns[1], s =50, marker='o')
    ax1.scatter(x = df.iloc[:,2],  y = y_range, color = 'red', label = df.columns[2], s = 50,marker = 'H')
    ax1.set_yticks(y_range) 
    ax1.set_yticklabels(df1.iloc[:,0])
    ax1.set_ylabel('police districts', fontsize = 14, color = 'gray')
    ax1.set_ylim(tot_recs+1,0)
    ax1.set_xlabel('Percent', fontsize = 14, color = 'gray')
    ax1.legend(fontsize = 12,loc='best')
    ax1.set_title('Police rec % < pop % \nBias (pop - pol_rec) ' + df.columns[1], fontsize = 14)
    plt.grid(False)
    ax1.set_facecolor('white')
    fig.set_facecolor('w')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)    
    plt.tight_layout()
    return


# In[ ]:


def utility_metadata_summary(file_path):
    ''' Function to summarise metadata file. The file has components of meta data seperated by _ in code and ; in description.
    It returns unique combinations of code and description part. This will help in grouping and extracting only needed columns
    from data.
    Input - Metadata file path.
    Output - Dataframe with code, desc and type unique combinations
    '''
    meta_df = pd.read_csv(file_path,names = ['code','desc'])
    meta_code = meta_df['code'].str.split('_',expand = True)
    meta_desc = meta_df['desc'].str.split(';',expand = True)
    temp_df = pd.DataFrame(columns = ['code','desc','type'])
    for cols in meta_code.columns:
        t = pd.concat([meta_code[cols], meta_desc[cols]], axis=1)
        t['type'] = 'type'+str(cols)
        t.columns = ['code','desc','type']
        temp_df = pd.concat([temp_df,t],axis=0)

    temp_df.drop_duplicates(inplace = True)
    temp_df.dropna(subset = ['code','desc'],inplace = True)
    temp_df.reset_index(inplace = True, drop = True)
    return(temp_df)


# # Dept_11_00091.  (Suffolk County, Massachusetts).
# *If you want to see how below conclusions reached, read on.  *  
# 
# *If you want to see the code written for this analysis click on the button **"Click here to toggle on/off the raw code"** above.*
# 
# ## Conclusions . 
# 
# 1. Police records for males are 7.5 time police records for females.  
# 2. 58.6% of total police records are for subject race black.  
# 3. Asian, midle-eastern/indin and native american (indian) police records are way too small compared to other races.  
# 4. Higher police records for Male group and black group overlap with overall high police records.  For 'white' group it is less clear. Highest police records for 'white' group are in regions where overall police record are highest in the region.  
# 5. From series of choropleths and correlation maps, total police records has strong postive correlation with black race population, education up to highschool only, poverty rate, and population with age up to 14.  
# 6. Police districts nicely cluster together into 2 group. Group 0 is C1, B2, B3, D4 and D14. Rest in group 1. Group0 is characterised by total police records with strong positive correlation with hispanic race and black race population, less education and population with age upto 14. Group1 is characterised by very strong positive correlation with asian race, renter occupied housing units.  
# 7. White, female, asian, native american groups have higher population % than police records % accross the police districts.  
# 8. For black and male groups almost always the police records % is more than population %.  
# 9. D14, E13 have higher % of investigative police records while a15 and c6 police districts have least % indicating there is scope of redployment of policing.
# 
# ## Execute step 1.  (step1_police_recs.)  
# Police records have 34 columns. 
# Police shapefiles have 4 columns. CRS set is 'epsg:2249'.  
# Analysis of column contents does not return perfcect common key. However 'DISTRICT' and 'DIST' seems to be good candidates for joining the datasets. Let's analyse further.
# 

# In[ ]:


folder_name = '../input/data-science-for-good/cpe-data/Dept_11-00091/'
(pg,pr) = step1_police_recs(folder_name)


# 'DISTRICT' and 'DIST' seems good candidates for common keys. Let's explore why the common rate is only 50%.  
# Though the common unique keys are 50%, 99.58% of police records are for these common keys only. Very small number less than 0.5% of records will be dropped once police_records  and police_shapefile dtaframes are joined on 'DIST' and 'DISTRICT.

# In[ ]:


pd.merge(pr,pg, left_on = 'DIST',right_on = 'DISTRICT',how = 'inner').DIST.count()/pr.DIST.count()*100


# ## Execute step 2. (step2_crime_summary)  
# 
# ### Preprocessing ...
# Analyse the police records data frame to determine which columns need to be used for summarization. One of the main problem statement is about racial bias. We may extend that to gender as well. So summary of police records by sex and race seems good starting point.  
# 
# **SEX** column. There are only 237 records with 'UNKNOWN'. We need to replace these with 'NaN' to be taken out for summarizing  by sex.

# In[ ]:


pr.SEX.value_counts()


# In[ ]:


pr.SEX.replace('UNKNOWN',np.NaN, inplace = True)
pr.SEX.value_counts()


# **SUBJECT RACE** -> To have consistent and clean values for summary naming race description is renamed.  
# Unknown and no data records are dropped. Below are the before and after clean up.

# In[ ]:


pr.DESCRIPTION.value_counts()


# In[ ]:


pr.DESCRIPTION.replace({'B(Black)' : 'black',
                       'W(White)':'white',
                       'H(Hispanic)':'hispanic',
                       'A(Asian or Pacific Islander)' : 'asian',
                       'M(Middle Eastern or East Indian)' : 'mideast_eastind',
                       'I(American Indian or Alaskan Native)' : 'native_indian',
                       'UNKNOWN' : np.NaN,
                       'NO DATA ENTERED' : np.NaN}, inplace = True)
pr.DESCRIPTION.value_counts()


# ### Execute step 2 after cleanup.  
# 
# After cleanin up race labels and identifying columns 'SEX' and 'DESCRIPTION' for summarization, pass these values to step2_crime_summary() function.

# In[ ]:


cols_summary = ['SEX','DESCRIPTION']
crimes_by_precinct = step2_crime_summary(pr,cols_summary,'DIST',pg,'DISTRICT')
crimes_by_precinct.head()


# ## Execute step 3. (choropleth_map)   
# 
# Visualize map of police districts with choropleth by each value of summary columns overlayed with bubble plot of total police records. To draw insight into police records hot zones and how they relate to police records by gender and race of subjects.

# In[ ]:


cols = crimes_by_precinct.iloc[:,3:].columns
choropleth_map(crimes_by_precinct,
               cols,
               bubble = 'total_police_records',
               lab_title='police records range',
               fsize = (16,20),
               color_map = 'Greens')


# ## Execute step4. (step4_merge_census_crimes)  
# 
#     1. The first plot gives a perspective of police precincts area under analysis with respect to overall state map.  
#     2. Second plot is only census tracks overlap with police precincts annotated with police dristricts.  
#     3. Below is the Police District and census track (AFFGEOID) and percent overlap dtaframe head. For e.g. this shows first few rows of A1 DISTRICT overlap with census tracks. '1400000US25025020101' has 95.3% overlap with A1.

# In[ ]:


cf_path = '../input/massachusetts-census-track'
pp_census_overlap = step4_merge_census_crimes(cf_path, crimes_by_precinct,'DISTRICT',tpath = '../input/temp-mass-merge/mass.shp')
pp_census_overlap.head()


# ### Execute step5 - (step5_acs_data_summary)  
#     
#     Pass the folder_path. Output is the acs survey data aggregated as specified above. Below are first few rows of the extracted ACS data.

# In[ ]:


acs_total_pop = step5_acs_data_summary(folder_name)
acs_total_pop.head()


# ###  Execute step 6 - (step6_merge_acs_pp_data).
#     
#     ACS survey data is extracted and computed for analsysis. In abvoe steps we had police records by police disticts calculated and also overlap of ACS tracks with police districts. Now the time merge these 2 together to get ACS and police records data by police district.

# In[ ]:


pp_acs_total_pop = step6_merge_acs_pp(pp_census_overlap, acs_total_pop)


# ### Execute step7 - (step7_append_geo).  
# 
# Append geospatial columns of police district to ACS calculated dataframe.

# In[ ]:


crimes_pp_total_pop = step7_append_geo(crimes_by_precinct,pp_acs_total_pop)


# ### Execute step 8 - (ste8_corr_plot).  
# 
# Get the coorelation plot of all ACS and total police records dataframe merged.

# In[ ]:


ste8_corr_plot(crimes_pp_total_pop)


# ### Execute step9 - (step9_elbow_plot).
# 
#     To get best cluster number based on elbow plot.

# In[ ]:


t22_scaled = step9_elbow_plot(crimes_pp_total_pop)


# ### Execute step 10 - (step10_cluster_and_corr).

# In[ ]:


temp = step10_cluster_and_corr(crimes_pp_total_pop, num_clust = 2)


# ### How population and police records compare?  
# 
# Compare percent of population and percent of police records for that group to get an indication of any bias?

# In[ ]:


step11_bias_calc(crimes_by_precinct, pp_acs_total_pop)


# ### Step 12. True Crimes? 
# 
# Police records for investigative purposes. Studying the police records, there are records which are investigative in nature. Below condition is used to categorize such records. Other records has a reason for search or it results in 
# 
# 1. STOP_REASON is blank or 'INVESTIGATIVE'
# 2. FIOFS_REASONS is 'INVESTIGATIVE.*'
# 3. BASIS is blank.

# In[ ]:


inv_str = re.compile('INVESTIGATE,*')
pr_investigative = pr[(pr.BASIS.isnull())&(((pr.STOP_REASONS == 'INVESTIGATIVE') | (pr.STOP_REASONS.isnull())) & (pr.FIOFS_REASONS.str.contains(inv_str)))]
pr_invest_df = pd.DataFrame(pr_investigative.DIST.value_counts()).reset_index()
pr_invest_df.columns = ['DISTRICT','investigative_counts']


# In[ ]:


inv_by_pp = crimes_by_precinct[['DISTRICT',
                                'geometry',
                                'total_police_records']].merge(pr_invest_df, 
                                                               on = 'DISTRICT', 
                                                               how = 'left')


# In[ ]:


inv_by_pp['inv_perc'] = inv_by_pp['investigative_counts']/inv_by_pp['total_police_records']*100
inv_by_pp.sort_values(by = 'inv_perc',ascending = False, inplace = True)
inv_by_pp


# In[ ]:


fig,(ax0,ax1) = plt.subplots(nrows = 1, ncols=2,figsize = (12,6))
yrange = np.arange(1,len(inv_by_pp)+1)
ax1.barh(y = yrange, width = inv_by_pp.inv_perc, color = 'grey')
ax1.set_yticks(yrange)
ax1.set_yticklabels(inv_by_pp['DISTRICT'])
ax1.set_title('Percent of total records investigative type', fontsize=14)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

inv_by_pp.plot(ax = ax0, facecolor = 'None',edgecolor = 'grey', figsize=(10,10))
ax0.scatter(x = inv_by_pp.geometry.centroid.x,
          y = inv_by_pp.geometry.centroid.y,
          s = inv_by_pp.total_police_records/10,
          alpha=0.4,
          color = 'r')
ax0.scatter(x = inv_by_pp.geometry.centroid.x,
          y = inv_by_pp.geometry.centroid.y,
          s = inv_by_pp.investigative_counts/10,
          alpha=0.4,
          color = 'b')
ax0.set_axis_off()
for i,pp in enumerate(inv_by_pp.DISTRICT):
    ax0.annotate(pp, (inv_by_pp.iloc[i].geometry.centroid.x, inv_by_pp.iloc[i].geometry.centroid.y),
               fontsize=14)
ax0.set_title('Tot police recs and investigative type', fontsize=14)    
plt.tight_layout()


# # Dept_37_00049 (Dallas County, Texas)  
# 
# Above same steps are for other dataset provided in Dept_37_00049. Results are below. It provides such a nice template and same patterns. Only pending is to get conclusions based on results !!!
# 
# ## Execute step 1.  (step1_police_recs.)  
# 
# Police records have 47 columns. 
# Police shapefiles have 4 columns. CRS set is datum:NAD83'.  
# 
# Analysis of column contents does not return perfcect common key. We need to use LATITUDE AND LONGITUDE columns and geospatial join to merge police records with police shapefiles.

# In[ ]:


folder_name = '../input/data-science-for-good/cpe-data/Dept_37-00049/'
(pg,pr) = step1_police_recs(folder_name)


# pg.Name is unique identifier in police shape files. It will be key yo merge and will show up on all charts etc. Lets's clean it up. Replace ' ' with '' and remove EPIC prefix from all names.  
# Reset the CRS to epsg:32118.

# In[ ]:


pg.head()


# In[ ]:


name_str = re.compile('EPIC.')
pg['Name'] = pg['Name'].str.replace(name_str,'')
pg['Name'] = pg['Name'].str.replace(' ','')
pg.to_crs({'init': 'epsg:32118'},inplace = True)
pg


# Create 'geometry' column for pr using LATITUDE and LONGITUDE. They are in decimal degrees. Meaning they are in epsg:4326. They will be converted to epsg:32118 before running geospatial join.

# In[ ]:


pr.dropna(subset = ['Longitude','Latitude'],inplace = True)
pr['geometry'] = list(zip(pr['Longitude'],pr['Latitude']))
pr['geometry'] = pr['geometry'].apply(Point)
pr = gpd.GeoDataFrame(pr,
                                 geometry = 'geometry',
                                 crs ={'init': 'epsg:4326'} )
pr.to_crs({'init': 'epsg:32118'},inplace = True)
ax = pg.plot(facecolor = 'None', edgecolor = 'black', figsize=(8,8))
pr.plot(ax=ax, alpha = 0.2)
ax.set_axis_off()


# This shows both police records and police geopandas are in same CRS and there are many records which don't lie within police district boundaries.

# In[ ]:


# Again kaggle has issues with geopandas spatial joins. So this wrokaround of uploading locally saved file. Below commented lines work like charm on local machine
#pr1 = gpd.sjoin(pr,pg, how = 'inner', op = 'intersects')
#pr1.drop(columns = ['index_right'],inplace = True)
#pr1.reset_index(drop=True,inplace = True)
pr1 = gpd.read_file("../input/temp-tx-sjoin/tx1.shp")
len(pr1)


# Only 237 out of 2300+ records fall within the boundaries of police shapefiles provided.

# In[ ]:


pr1.CitRace.value_counts()


# In[ ]:


pr1.CitSex.value_counts()


# In[ ]:


pr1.CitSex.replace({'Male':'MALE',
                   'Female': 'FEMALE'},inplace = True)
pr1.CitRace.replace({'Black':'black',
                   'Hispanic': 'hispanic',
                   'White':'white'},inplace = True)
pr1.head()


# In[ ]:


pg.reset_index(drop=True,inplace = True)


# ### Execute step 2 after cleanup.  
# 
# After cleanin up race labels and identifying columns 'CitSex' and 'CitRace' for summarization, pass these values to step2_crime_summary() function.

# In[ ]:


cols_summary = ['CitSex','CitRace']
crimes_by_precinct = step2_crime_summary(pr_df = pr1,pr_cols_summary = cols_summary,pr_common_key = 'Name',pg_df = pg,pg_common_key = 'Name')
crimes_by_precinct.head()


# ## Execute step 3. (choropleth_map)   
# 
# Visualize map of police districts with choropleth by each value of summary columns overlayed with bubble plot of total police records. To draw insight into police records hot zones and how they relate to police records by gender and race of subjects.

# In[ ]:


cols = crimes_by_precinct.iloc[:,3:].columns
choropleth_map(crimes_by_precinct,
               cols,
               bubble = 'total_police_records',
               lab_title='police records range',
               fsize = (16,20),
               color_map = 'Greens',
              size_divider = 0.05)


# ## Execute step4. (step4_merge_census_crimes)  
# 
#     1. The first plot gives a perspective of police precincts area under analysis with respect to overall state map.  
#     2. Second plot is only census tracks overlap with police precincts annotated with police dristricts.  

# In[ ]:


aff_path = '../input/data-science-for-good/cpe-data/Dept_37-00049/37-00049_ACS_data/37-00049_ACS_education-attainment/ACS_15_5YR_S1501_with_ann.csv'
AFFGEOID_list = list(pd.read_csv(aff_path,skiprows = [1])['GEO.id'])

cf_path = '../input/texas-census-tracks'
pp_census_overlap = step4_merge_census_crimes(cf_path, crimes_by_precinct,'Name',list_id = AFFGEOID_list, tpath = '../input/temp-tx-merge/tx2.shp')
pp_census_overlap.head()


# ### Execute step5 - (step5_acs_data_summary)  
#     
#     Pass the folder_path. Output is the acs survey data aggregated as specified above. Below are first few rows of the extracted ACS data.

# In[ ]:


acs_total_pop = step5_acs_data_summary(folder_name)
acs_total_pop.head()


# ###  Execute step 6 - (step6_merge_acs_pp_data).
#     
#     ACS survey data is extracted and computed for analsysis. In abvoe steps we had police records by police disticts calculated and also overlap of ACS tracks with police districts. Now the time merge these 2 together to get ACS and police records data by police district.

# In[ ]:


pp_acs_total_pop = step6_merge_acs_pp(pp_census_overlap, acs_total_pop, common_key = 'Name')


# ### Execute step7 - (step7_append_geo).  
# 
# Append geospatial columns of police district to ACS calculated dataframe.

# In[ ]:


crimes_pp_total_pop = step7_append_geo(crimes_by_precinct,pp_acs_total_pop, common_key = 'Name', size_divider = 0.1)


# ### Execute step 8 - (ste8_corr_plot).  
# 
# Get the coorelation plot of all ACS and total police records dataframe merged.

# In[ ]:


ste8_corr_plot(crimes_pp_total_pop)


# ### Execute step9 - (step9_elbow_plot).
# 
#     To get best cluster number based on elbow plot.  
#     AS there are only 6 police departments. cluster size of 2 is selected to get better segregated clusters. Else clusters may be individual pds.

# In[ ]:


t22_scaled = step9_elbow_plot(crimes_pp_total_pop, max_size = 5)


# ### Execute step 10 - (step10_cluster_and_corr).  
# 
# As there are only 6 police departments, cluster size of 2 is selected. One cluster has 2 pd. do correlation is 1 or -1 !!
# Can't get more insights.
# 

# In[ ]:


temp = step10_cluster_and_corr(crimes_pp_total_pop, num_clust = 2, common_key = 'Name')


# ### How population and police records compare?  
# 
# Compare percent of population and percent of police records for that group to get an indication of any bias?

# In[ ]:


crimes_by_precinct['asian'] = 0
crimes_by_precinct['native_indian'] = 0
step11_bias_calc(crimes_by_precinct, pp_acs_total_pop, common_key = 'Name')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




