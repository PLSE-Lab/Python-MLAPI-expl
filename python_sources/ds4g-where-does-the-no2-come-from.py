#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import scipy

import matplotlib.pylab as plt
import seaborn as sns

import rasterio as rio
from rasterio.warp import reproject, Resampling

import os


# # The goal of this challenge is to: 
# 
# > Develop a methodology to calculate an average historical emissions factor of electricity generated for a sub-national region, using remote sensing data and techniques.
# 
# 
# ### 1. What is a emission factor (EF)?
# 
# In short, is a number that converts human activities to amount of pollutant released due to that activity. In this challenge, the focus is on trying to relate energy produced with NO$_2$ released:
# 
# NO$_2$ = EF * Energy_created - Other_Factors
# 
# The exact units of EF depend on the units of the NO$_2$ and the energy used. I'll use EF in **  10$^3$ kg NO$_2$/electrical generating megawatts /year**.
# 
# 
# 
# ### 2. Why is is hard to measure and how could it be easier?
# 
# From the challenge description says:
# 
# > Current emissions factors methodologies are based on time-consuming data collection and may include errors derived from a lack of access to granular datasets, inability to refresh data on a frequent basis, overly general modeling assumptions, and inaccurate reporting of emissions sources like fuel consumption.
# 
# If we could use satelitte images, particularly of some pollutants, and 'trace them back' to what produced them (for example a power plant), then we could have a handy measurement of the emission factors, for some properties of the power plant (fuel type, energy capacity).
# 
# 
# 
# ### 3. Summary of this analysis
# 
# Here I'm using the satelite images of NO$_2$ column density and 'distribute' that NO$_2$ amongst the power plants of Puerto Rico, using a (rather naive) assumption that the closest a pixel is of a power plant, the more the NO$_2$ on that pixel should have been produced by that power plant, but some of it might have still be produced by some closer by power plant. More on this lower down.
# 
# I'll also add some other data sets privided in the challenge, that should try to account for either other sources of NO$_2$ production (such as people travelling around) or the fact that the NO$_2$ may move around due to winds, rain, etc.
# 
# Although the actual electricity produced by a power plant should be a more precise way of measuring the EF, I'll use the electrical generating capacity in megawatts in the provided data set, that is more easily available.
# 
# I'll also focus on power plants that use **Oil** as their primary fuel, although in principle I could have used the same analysis for other kinds of power plants.
# 
# When calculating the EF, I'll use a **Hierarchical Bayesian** simple model, which has the advatage of using all the data of all Oil power plants to calculate an yearly average EF for these power plants, while still accounting for the fact that each oil power plant will have a slightly different EF due to other properties (year of construction, average use) that may even not be known.
# 
# 
# 
# ### 4. Outlook
# 
# [1. Power Plants](#intLink_power_plant)
# 
# [2. Satelite data](#intLink_2d)
#   
# [3. Splitting NO$_2$ between (overlapping) the power plants](#intLink_wmaps)
# 
# [4. EF calculation: Bayesian approach](#intLink_bayes)

# <div id="intLink_power_plant">
# # 1. Power Plants
#         
# This is a quick exploration and cleaning of the power plants in Puerto Rico data set, that I'm using to estimate the Emission Factors.

# In[ ]:


df_power_plants = pd.read_csv('../input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')


# In[ ]:


def make_summary_pd(df):
    df_summary = pd.DataFrame()
    df_summary['dtype'] = df.dtypes
    df_summary['NaNs'] = df.isna().sum()
    df_summary['unique_values'] = df.nunique()
    df_summary['min'] = df.min()
    df_summary['max'] = df.max()
    df_summary['mean'] = df.mean()
    df_summary['std'] = df.std()
    return df_summary

df_power_plant_quality = make_summary_pd(df_power_plants)
df_power_plant_quality.head(50)


# Filter variables that have only nans or the same value everywhere

# In[ ]:


bad_variables = df_power_plant_quality[df_power_plant_quality.unique_values < 2].index.tolist()
print('Removed variables',bad_variables)
df_power_plants.drop(bad_variables,axis=1,inplace=True)


# Extract longigute and latitute from .geo column

# In[ ]:


import json

def string_to_dict(dict_string):
    # Convert to proper json format (from here: https://stackoverflow.com/questions/39169718/convert-string-to-dict-then-access-keyvalues-how-to-access-data-in-a-class)
    dict_string = dict_string.replace("'", '"').replace('u"', '"')
    return json.loads(dict_string)['coordinates']

df_power_plants['coord'] = df_power_plants['.geo'].apply(string_to_dict)


# Plot some of the most interesting variables:

# In[ ]:


fig, ax = plt.subplots(1,5,figsize=(15,5))
fig.subplots_adjust(bottom=0.4,left=0.05,right=0.99)
sns.distplot(df_power_plants.capacity_mw.to_numpy(),ax=ax[0])
sns.distplot(df_power_plants.estimated_generation_gwh,ax=ax[1])
sns.countplot(df_power_plants.commissioning_year,ax=ax[2])
sns.countplot(df_power_plants.primary_fuel,ax=ax[3])
sns.countplot(df_power_plants.owner,ax=ax[4])

dd = [x.tick_params(axis='x', rotation=85) for x in ax[1:]]


# On a first approach, I'd say the most distinguishing factors of power plants are the *primary fuel* and the *capacity_mw* (the electrical generating capacity in megawatts) or *estimated generation*, both measures of how much electricity these power plants can produce.
# 
# It might be useful to see really how much these 3 factors vary accross the power plants, and if they are correlated.

# In[ ]:


fig, ax = plt.subplots(1,3,figsize=(15,5))
fig.subplots_adjust(bottom=0.2,left=0.05,right=0.99)

sns.boxplot(x="primary_fuel", y="capacity_mw", data=df_power_plants,ax=ax[0])
sns.boxplot(x="primary_fuel", y="estimated_generation_gwh", data=df_power_plants,ax=ax[1])
sns.scatterplot(x="capacity_mw", y="estimated_generation_gwh", hue="primary_fuel",  data=df_power_plants,ax=ax[2])
ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[2].set_yscale('log')


# Fist plot: It seems that power plants running on *Oil have usually a higher capacity*, although notice that there is a large overlap with the capacity of plants running on other primary fuels. Oil just seems to be more flexible.
# 
# Second plot: A little the same as for the first, so are these two quantities correlated? That's what we're inspecting in the:
# 
# Third plot: (y is in log scale) there seems to be a correlation between estimated generation and capacity, but only for power plants running on Oil and Gas.

# <div id="intLink_2d">
# ## 2. Satelite data 
#     
# I'll use 3 data sets:
# 
# * The [Sentinel 5P](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2), which contains the NO$_2$ observations besides some others (see bellow). Resolution 0.01 arc degrees.
# * The [Global Forecast System](https://developers.google.com/earth-engine/datasets/catalog/NOAA_GFS0P25) that contains a lot of interesting weather observations (temperature, precipitation...). Resolution  0.25 arc degrees
# * The population density of Puerto Rico, originally from [GPWv411:](https://developers.google.com/earth-engine/datasets/catalog/CIESIN_GPWv411_GPW_Basic_Demographic_Characteristics), but you can find a dataset for Puerto Rico in kaggle [here](https://www.kaggle.com/vpatricio/population). Resolution 30 arc seconds --> 0.0083 arc degreees
# 
# 
# First I'll just pick an image and check what data we have available in each band for each data set.

# **Sentinel 5P**

# In[ ]:


image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20190418T165428_20190424T183227.tif'
im = rio.open(image)
im.descriptions


# I'll mainly focus on the five first. The last 5 are mostly data relative to the sensor, so I don't think they'll be very useful.

# In[ ]:


sentinel_bands = {1: 'NO2_column_number_density',
                  2: 'tropospheric_NO2_column_number_density',
                  3: 'stratospheric_NO2_column_number_density',
                  5: 'tropopause_pressure',
                  6: 'absorbing_aerosol_index',
                  7: 'cloud_fraction'}


# **GFS**

# In[ ]:


image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/gfs_2018093006.tif'
im = rio.open(image)
im.descriptions


# I'm keeping most of them for now.

# In[ ]:


gfs_bands = {1: 'temperature_2m_above_ground',
             3: 'relative_humidity_2m_above_ground',
             4: 'u_component_of_wind_10m_above_ground',
             5: 'v_component_of_wind_10m_above_ground',
             6: 'precipitable_water_entire_atmosphere'}


# And I'll also use the population density map. 

# ### 2.1 Some boring pre-processing
# 
# Since the GFS is the data set with the lowest resolution (0.25 arc degrees), I'll regrid all the other images to this one, to avoid interpolations. (note: I am not totally sure if the images had already been regrided to the same pixel size.)
# 
# Also, I'll also *keep only one image per day* both for the GFS and the Sentinel data sets, arbitrarily keeping the first, to simplify things. 
# 
# (This takes a couple of minutes and can probably be improved...)

# In[ ]:


im_ref_path = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/gfs_2018093006.tif'


# In[ ]:


def images_path_and_date(dataset):
    
    global_path = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/'
    
    if dataset == 's5p_no2':
        data_path = global_path+'s5p_no2'
        data_prefix = 's5p_no2_'
        file_list = os.listdir(data_path)
        dates = [pd.to_datetime(file.split('s5p_no2_')[1].strip('.tif').split('_')[0]) for file in file_list]    
        
    if dataset == 'gldas':
        data_path = global_path+'gldas'
        file_list = os.listdir(data_path)
        dates = [pd.to_datetime(file.strip('gldas_').strip('.tif').replace('_','T')) for file in file_list]
        
    if dataset == 'gfs':
        data_path =  global_path+'gfs'
        file_list = os.listdir(data_path)
        dates = [pd.to_datetime(file.strip('gfs_').strip('.tif')[:8]) for file in file_list]
        
    df = pd.DataFrame(data={'file':file_list,'date':dates}).sort_values('date')  
    
    # Keep only one file per day
    df['date'] = df.date.dt.date
    df.drop_duplicates('date',keep='first',inplace=True)
    
    return df


def load_and_register_one_image(im_path,full_path,band,gfs):
    
    if gfs:
        return rio.open(full_path+im_path).read(band)
    
    else:
        ref_im = rio.open(im_ref_path)
        destination = np.zeros_like(ref_im.read(1))
        ref_transform = ref_im.transform
        ref_crs = ref_im.crs
        im_src =  rio.open(full_path+im_path)
        return reproject(
                    source = im_src.read(band),
                    destination = destination,
                    src_transform = im_src.transform,
                    src_crs = im_src.crs,
                    dst_transform = ref_transform,
                    dst_crs = ref_crs,
                    resampling = Resampling.nearest
                    )[0]
    

def register_all_images(full_path,band,files,gfs=False):
    return [load_and_register_one_image(im_path,full_path, band=band,gfs=gfs) for im_path in files]


# In[ ]:


df_s5p_files = images_path_and_date(dataset='s5p_no2')
df_gfs_files_all = images_path_and_date(dataset='gfs')
df_gfs_files = pd.merge(df_gfs_files_all,
                        df_s5p_files['date'],
                        left_on='date',
                        right_on='date',
                        how = 'inner'
                        ) 


# In[ ]:


full_path_s5p = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/'
for band,name in sentinel_bands.items():
    df_s5p_files[name] = register_all_images(full_path = full_path_s5p,
                                             band=band,
                                             files = df_s5p_files.file)


# In[ ]:


full_path_gfs = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/'
for band, name in gfs_bands.items():
    df_gfs_files[name] = register_all_images(full_path = full_path_gfs,
                                               band=band,
                                               files = df_gfs_files.file,
                                               gfs=True)


# In[ ]:


pop_density = load_and_register_one_image(im_path = 'GPWv411_Puerto_Rico_population_density_map.tif',
                                          full_path = '/kaggle/input/population/',
                                          band = 1,
                                          gfs = False)


# Yearly means (full average of all images available for each data set):

# In[ ]:


def mean_image(x):
    return np.nanmean(np.dstack(x),axis=2).tolist()

mean_NO2_col_density = mean_image(df_s5p_files.NO2_column_number_density)
mean_tropo_NO2 = mean_image(df_s5p_files.tropospheric_NO2_column_number_density)
mean_strato_NO2 = mean_image(df_s5p_files.stratospheric_NO2_column_number_density)
mean_abs_ind = mean_image(df_s5p_files.absorbing_aerosol_index)
mean_tropo_pause = mean_image(df_s5p_files.tropopause_pressure)
mean_cloud_fraction = mean_image(df_s5p_files.cloud_fraction)

mean_temperature = mean_image(df_gfs_files.temperature_2m_above_ground)
mean_humidity = mean_image(df_gfs_files.relative_humidity_2m_above_ground)
mean_u_wind = mean_image(df_gfs_files.u_component_of_wind_10m_above_ground)
mean_v_wind = mean_image(df_gfs_files.v_component_of_wind_10m_above_ground)
mean_precip_water = mean_image(df_gfs_files.precipitable_water_entire_atmosphere)


# ### 2.2 Finally, some plots!

# In[ ]:


def plot_power_plants(ax,impath=im_ref_path):
    
    color_dict = {'Hydro':'white',
                 'Oil':'C3',
                 'Solar' : 'gold',
                 'Gas' : 'darkorange',
                 'Coal' : 'khaki',
                 'Wind' : 'deepskyblue'}
    
    marker_dict = {'Hydro':'p',
                  'Oil':'X',
                  'Solar' : '>',
                  'Gas' : 'v',
                  'Coal' : '^',
                  'Wind' : '<'}
    
    im = rio.open(impath)
    
    for fuel,_ in color_dict.items():
        ax.plot(-10,10,marker=marker_dict[fuel],color=color_dict[fuel],label=fuel,linestyle='')

    for i,(coord,fuel,cap) in enumerate(zip(df_power_plants.coord,df_power_plants.primary_fuel,df_power_plants.capacity_mw,)):
        lon,lat = coord
        row,col = im.index(lon,lat)
        ax.plot(col,row,marker=marker_dict[fuel],color=color_dict[fuel],markersize=10*np.log(cap)/5,alpha=0.7)
        if fuel == 'Oil':
            ax.annotate(df_power_plants.iloc[i]['name'],xy=(col+7,row),color='C3',fontsize=10,weight='bold')


# In[ ]:


fig, ax = plt.subplots(3,1, figsize=(16,12))
ax = ax.ravel()
sns.heatmap(mean_NO2_col_density,ax=ax[0],square=True,cmap='viridis')
sns.heatmap(mean_tropo_NO2,ax=ax[1],square=True,cmap='viridis')
sns.heatmap(mean_strato_NO2,ax=ax[2],square=True,cmap='viridis')
[x.axis('off') for x in ax]
[plot_power_plants(x) for x in ax]

ax[0].set_title('Total NO$_2$ column number density')
ax[1].set_title('tropospheric NO$_2$  column number density')
ax[2].set_title('stratospheric NO$_2$  column number density')
plt.legend();


# These are maps of NO$_2$ column density. The units of the maps are in mol/m$^2$, so quantity of NO$_2$ molecules per surface.
# 
# I'm also plotting the position of the power plants with dots in different colours and symbols, depending on the primary fuel, with the size depending on the electrical generating capacity in megawatts. In the following section I'll focus on the Oil power plants, the ones with big red crosses and with the name attached.
# 
# Although the densities of NO$_2$ are different in maps of total density (top) and tropospheric densities (middle), these have very similar patterns. Once it gets to the stratosphere things get messier. Since we want to somehow correlate power plants with NO$_2$, measurements closer to the ground (so trosposphere) will probably give us a better signal.
# 
# There are also other interesting bands:

# In[ ]:


fig, ax = plt.subplots(4,2, figsize=(20,12))
ax = ax.ravel()
fig.subplots_adjust(wspace=0.1)

sns.heatmap(mean_abs_ind,ax=ax[0],square=True,cmap='Greys')
sns.heatmap(mean_tropo_pause,ax=ax[1],square=True,cmap='Greys',vmax=7800)
sns.heatmap(mean_cloud_fraction,ax=ax[2],square=True,cmap='Greys')
sns.heatmap(mean_temperature,ax=ax[3],square=True,cmap='Greys',vmax=32)
sns.heatmap(mean_humidity,ax=ax[4],square=True,cmap='Greys',vmax=100)
sns.heatmap(mean_u_wind,ax=ax[5],square=True,cmap='Greys',vmax=5,vmin=-5)
sns.heatmap(mean_v_wind,ax=ax[6],square=True,cmap='Greys',vmax=5,vmin=-5)
sns.heatmap(mean_precip_water,ax=ax[7],square=True,cmap='Greys',vmax=40)
[x.axis('off') for x in ax]
[x.contour(mean_NO2_col_density, cmap='viridis') for x in ax]
    

ax[0].set_title('absorbing aerosol index')
ax[1].set_title('tropopause pressure')
ax[2].set_title('cloud fraction')
ax[3].set_title('temperature')
ax[4].set_title('relative humidity')
ax[5].set_title('wind u component')
ax[6].set_title('wind v component')
ax[7].set_title('precipitation');


# For example, the [absorbing aerosol index](https://en.wikipedia.org/wiki/Aerosol) (basically maps stuff in the air, can be natural - like fog or dust - or human made - like pollutants) correlates somewhat with the NO$_2$ total density (in contours). Not sure if NO$_2$ is used when calculating the index, in that case it should not be used to predict NO$_2$ density. However it might also just flag places where *general stuff in the air* gets trapped, like valleys or cities, and the NO$_2$ migh also concentrate there, even if it is produced elsewhere.
# 
# Cloud fraction also seems to have some correlation with NO$_2$ density, although the rest of the properties seem to be mapped at much lower resolution, so it's hard to tell. 
# 
# I'm calculating the [spearman correlation rank](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) of each pixel of these maps and see how the correlate with NO$_2$ density.

# In[ ]:


print('Correlatio with NO2:\n')
print('Tropo Pause\t\t{:f}'.format(scipy.stats.spearmanr(np.ravel(mean_NO2_col_density),np.ravel(mean_tropo_pause)).correlation))
print('Tropospheric NO2\t{:f}'.format(scipy.stats.spearmanr(np.ravel(mean_NO2_col_density),np.ravel(mean_tropo_NO2)).correlation))
print('Stratospheric NO2\t{:f}'.format(scipy.stats.spearmanr(np.ravel(mean_NO2_col_density),np.ravel(mean_strato_NO2)).correlation))
print('Cloud fraction\t\t{:f}'.format(scipy.stats.spearmanr(np.ravel(mean_NO2_col_density),np.ravel(mean_cloud_fraction)).correlation))
print('Absorp. aerosol index\t{:f}'.format(scipy.stats.spearmanr(np.ravel(mean_NO2_col_density),np.ravel(mean_abs_ind)).correlation))
print('Temperature\t\t{:f}'.format(scipy.stats.spearmanr(np.ravel(mean_NO2_col_density),np.ravel(mean_temperature)).correlation))
print('Humidity\t\t{:f}'.format(scipy.stats.spearmanr(np.ravel(mean_NO2_col_density),np.ravel(mean_humidity)).correlation))
print('U wind\t\t\t{:f}'.format(scipy.stats.spearmanr(np.ravel(mean_NO2_col_density),np.ravel(mean_u_wind)).correlation))
print('V wind\t\t\t{:f}'.format(scipy.stats.spearmanr(np.ravel(mean_NO2_col_density),np.ravel(mean_v_wind)).correlation))
print('Precipitation\t\t{:f}'.format(scipy.stats.spearmanr(np.ravel(mean_NO2_col_density),np.ravel(mean_precip_water)).correlation))


# All correlations are pretty week, expect between total NO$_2$ and NO$_2$ in the troposphere and stratosphere, unsuprisingly, although *tropopause presure* and *cloud fraction* do seem to be somewhat correlation with total NO$_2$ column density.
# 
# Now let's go back to plant properties and map them, which will involve some experimentation. First i'll just make a blank map, go to the pixel where there is a power plan, and put their energy capacity on that pixel.

# In[ ]:


def map_power_plant_property(col):
    
    map_property = np.zeros(shape=(148, 475))
    im = rio.open(im_ref_path)
    
    for coord,value in zip(df_power_plants.coord,df_power_plants[col]):
        lon,lat = coord
        row,col = im.index(lon,lat)
        map_property[row,col] = value
        
    return map_property

capacity_map = map_power_plant_property('capacity_mw')
estimated_gen_map = map_power_plant_property('estimated_generation_gwh')


# In[ ]:


fig, ax = plt.subplots(2,2, figsize=(20,6))
ax = ax.ravel()
sns.heatmap(capacity_map,ax=ax[0],square=True,cmap='Reds',cbar_kws={'shrink':0.7})
conv_capacity_map = scipy.ndimage.gaussian_filter(capacity_map, sigma=10,)
sns.heatmap(conv_capacity_map,ax=ax[1],square=True,cmap='Reds',cbar_kws={'shrink':0.7})
sns.heatmap(mean_NO2_col_density,ax=ax[2],square=True,cmap='viridis',cbar_kws={'shrink':0.7})
sns.heatmap(mean_NO2_col_density,ax=ax[3],square=True,cmap='viridis',cbar_kws={'shrink':0.7})
sns.heatmap(conv_capacity_map,ax=ax[3],square=True,cmap='Reds',alpha=0.1,cbar=False)
[x.axis('off') for x in ax]
plot_power_plants(ax[2])

ax[0].set_title('Power Plant capacity (all in one pixel per power plant)')
ax[1].set_title('Power Plant capacity (convolved with an arbrirary 2D gaussian filter)')
ax[2].set_title('NO2 column number density')
ax[3].set_title('NO2 column number density + Convolved Power Plant Capacity');

ax[1].annotate('B',xy=(0.58,0.85),color='k',xycoords='axes fraction')
ax[1].annotate('A',xy=(0.61,0.1),color='k',xycoords='axes fraction')
plt.legend();


# Even if you can't see it, the top left map is not empty, but since the power plants are very tiny compared with the size of Puerto Rico, only a couple of pixels have values different from zero for energy capacity. 
# 
# I'm convolving the energy capacity map with a 2D gaussian filter, at the moment more for visualisation purposes than anything else.
# 
# If we overlay both maps, we can see that there is something else contributing to the NO$_2$ density in the air, because, for example, the two strongest 'poles' of electricity capacity -- I've called them A and B -- have very different NO$_2$ densities above them. Several options for this:
# 
# 0. (The power plant data set is not correct. That seems to be the case for the electrical generation, from a couple of posts, not sure about the electrical capacity) 
# 1. Differences on the properties of the power plants. Some are way more pollutent for the same amount of energy produced.
# 2. Other sources of NO$_2$. Apparently, [trafic is a major one.](https://www.kaggle.com/c/ds4g-environmental-insights-explorer/discussion/129991)
# 3. NO$_2$ may travel around a lot (due to weather and geography?), and we loose the spatial correlation between power plant and NO$_2$ in its vicinity.
# 
# I don't think number 1 explains most of the differences we see, since A and B have both a mixture of Oil and Gas. I can't do much about number 0, so I'll focus on the other two.

# <div id="intLink_2d_pop">
# ### 2.1 Population density
# 
# 
# For that I'll use another data set available in the Earth Engine Data Catalog, the [GPWv411: Basic Demographic Characteristics](https://developers.google.com/earth-engine/datasets/catalog/CIESIN_GPWv411_GPW_Basic_Demographic_Characteristics).
# 
# I had a look at this [kaggle notebook](https://www.kaggle.com/vpatricio/how-to-get-started-with-the-earth-engine-data/edit) and this [colab](https://colab.research.google.com/github/google/earthengine-api/blob/master/python/examples/ipynb/ee-api-colab-setup.ipynb) for this part.
# 
# (Note: because I didn't know what I was doing, I had to save the tif file on Google Drive and then upload it here (although it comes from a public data set), but it ended up with the lovelly name of imageToDriveExample.tif. Oh well, live and learn...)

# In[ ]:


'''from kaggle_secrets import UserSecretsClient
from google.oauth2.credentials import Credentials
import ee

user_secret = "vera_earth" 
refresh_token = UserSecretsClient().get_secret(user_secret)
credentials = Credentials(
        None,
        refresh_token=refresh_token,
        token_uri=ee.oauth.TOKEN_URI,
        client_id=ee.oauth.CLIENT_ID,
        client_secret=ee.oauth.CLIENT_SECRET,
        scopes=ee.oauth.SCOPES)
ee.Initialize(credentials=credentials)

# Puerto Rico 'geography', from the Official Thread
pr = ee.Geometry.Polygon(
        [[[-67.32297404549217, 18.563112930177304],
          [-67.32297404549217, 17.903121359128956],
          [-65.19437297127342, 17.903121359128956],
          [-65.19437297127342, 18.563112930177304]]], None, False);

dataset = ee.ImageCollection("CIESIN/GPWv411/GPW_Basic_Demographic_Characteristics").first()
raster = dataset.select('basic_demographic_characteristics');
export = ee.batch.Export.image.toDrive(image = raster,
                                       description = 'imageToDriveExample',
                                       region = pr,
                                         );
export.start()'''


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(14,4))
sns.heatmap(pop_density,ax=ax,square=True,cmap='cividis',cbar_kws={'shrink':0.7},vmax=200)
plot_power_plants(ax)
ax.axis('off');


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(14,6))
sns.heatmap(mean_NO2_col_density,ax=ax,square=True,cmap='viridis',cbar=False)
sns.heatmap(pop_density,ax=ax,square=True,cmap='Greys',alpha=0.2,vmax=100,cbar=False)
plot_power_plants(ax)
ax.axis('off');


# Great!
# 
# We can see that the highest NO$_2$ densities -- the bright yellow cloud on the top -- are found close (although not exctly on top of) the biggest city in Puerto Rico - San Juan.
# 
# Let's move to some analysis.

# <div id="intLink_wmaps">
# # 3. Splitting the NO$_2$ between power plants
# 
# As we see in the images above, one annoying thing about this problem is that power plants are tiny and the NO$_2$ each of them is producing can be spread over many kilometers. And we need to connect the two to calculate the emission factors.
# 
# One way to do this would be to measure the NO$_2$ density in a radius around each power plant and assume that that power plant is responsible for producing that NO$_2$. This is fine, but a bit of a problem when you have two or more power plants close by, since you'd count the NO$_2$ twice or more. 
# 
# The idea I'm using here is to split the NO$_2$ between power plants when they are close. If we have two power plants and we are 1 meter away from power plant A, probably 99.99% of the NO$_2$ comes from that plant, but if we are in between the two power plants then maybe it will closer to 50% of the NO$_2$. A caviat of this is that if we know that power plant A generates 10% more energy than the other, probably it will be responsible by more than 50% of the NO$_2$ that we can find at the mid-point of the two power plants, so that is something that needs to be improved in this analysis. One could also factor in the differente electric capacities of these power plants, but since this is a factor in the calculation of the EF (NO$_2$/electric capacity), the problem would be degenerate, so I dropped that idea. In short, I'm splitting equaly the NO$_2$ between power plants only based on distance, which is a strong assumption. 
# 
# So, the simple approach I'm using where is to say that the **NO$_2$ in each pixel is to be divided between *all* power plants weighted by the inverse of their distance to that point.**
# 
# (On a side note, the convolution I used above for plotting the 'average electical power' in the maps, might be a nicer alternative to the distance maps I used down below on this notebook, but the kernel size has to be set smartly, probably depending on how much the NO$_2$ dilutes on average and some knowledge on how the atmosphere works -- which I don't have... --  is needed here.)

# In[ ]:


def coord_to_xy(coord,im_ref_path):
    
    lon,lat = coord
    im_ref = rio.open(im_ref_path)
    row,col = im_ref.index(lon,lat)
    
    return (row,col)

def calculate_distance_map(coord,im,im_ref_path=im_ref_path):
    
    im = np.array(im)
    row,col = coord_to_xy(coord,im_ref_path)
    
    index_image = np.mgrid[:im.shape[0], :im.shape[1]].reshape(2,-1)
    centre = np.array((row,col)).reshape(2,1)
    
    dist = scipy.spatial.distance.cdist(centre.T,index_image.T)
    dist = np.reshape(dist,im.shape)
    
    return dist

def calculate_weight_maps(dist_maps):
    
    w_maps = [1/(dist+1) for dist in dist_maps]
    w_maps_fraction = w_maps / np.nansum(w_maps,axis=0)
    w_maps_fraction_norm =  w_maps_fraction / np.nansum(w_maps_fraction,axis=0)
    
    return w_maps_fraction_norm


def weighted_measure_around_power_plant(w_map,im,sumall=True):
    
    weighted_im = im*w_map
    
    if sumall:
        return np.nansum(weighted_im)
    else:
        return weighted_im


# In[ ]:


dist_maps = [calculate_distance_map(coord,mean_NO2_col_density,im_ref_path) for coord in df_power_plants.coord]
weight_maps = calculate_weight_maps(dist_maps)
weighted_no2_images = [weighted_measure_around_power_plant(w,mean_NO2_col_density,sumall=False) for w in weight_maps]
recovered_im = np.nansum(weighted_no2_images,axis=0)


# Hopefully this is a bit clerer in these plots bellow.
# 
# On the left I'm plotting the distance (top) and the NO$_2$ *weighted* NO$_2$ density map for one power plant, and on the right for other power plant.

# In[ ]:


fig, ax = plt.subplots(2,2,figsize=(20,4))
ax = ax.ravel()

sns.heatmap(dist_maps[0],ax=ax[0],square=True,cmap='magma')
sns.heatmap(dist_maps[1],ax=ax[1],square=True,cmap='magma')
sns.heatmap(weighted_no2_images[0],ax=ax[2],square=True,cmap='viridis')
sns.heatmap(weighted_no2_images[1],ax=ax[3],square=True,cmap='viridis')

ax[0].set_title('Distance (in pixels) map of power plant 1')
ax[1].set_title('Distance (in pixels) map of power plant 2')
ax[2].set_title('Weighting map of power plant 1')
ax[3].set_title('Weighting map of power plant 2')

d = [x.axis('off') for x in ax]


# If we sum all the weighted NO$_2$ density images, we recover the original yearly average one, which means all the NO$_2$ is distributed between the power plants and none is counted twice or more.

# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(20,4))
ax = ax.ravel()
ax[0].set_title('Mean NO$_2$ density')
ax[1].set_title('Sum of all the images for each power plant')
sns.heatmap(mean_NO2_col_density,ax=ax[0],square=True,cmap='viridis')
sns.heatmap(recovered_im,ax=ax[1],square=True,cmap='viridis')
d = [x.axis('off') for x in ax]


# <div id="intLink_bayes">
# # 4. Calculate the EF using a Bayesian approach
#   
# I'll use a very simple, linear model:
# 
# > NO$_2$ = EF $\times$ electrical_capacity + noise
# 
# And then try to 'unfold' that noise component by doing something like
# 
# > NO$_2$ = EF $\times$ electrical_capacity + K$_0$ $\times$ population + K$_1$ $\times$ temperature + K$_2$ $\times$ temperature ... + noise
# 
# It is very simple model, but very easy to intrepret as well.

# ### 4.1 Data
# 
# First let's just extract the data. The NO$_2$ here will be the daily average of the NO$_2$ density map, weighted by the distance maps, for each power plant.
# 
# Something that seems quite plausible is that the EF for Oil power plants should be quite different from the EF of hydro power plants, for example, since the way the energy is produced is quite different. So I'll just focus on the power plants that run primarily on Oil (just because there's a couple of them, with quite different electric capacities and close by populations, the same could be done for Hydro, Gas...).

# In[ ]:


oil_idx = df_power_plants[df_power_plants.primary_fuel=='Oil'].index
names = df_power_plants.iloc[oil_idx].name
capacities = df_power_plants.iloc[oil_idx].capacity_mw
dist_maps_oil = [dist_maps[i] for i in oil_idx]
weight_maps_oil = [weight_maps[i] for i in oil_idx]
#I'm adding the population in 10.000, just to bring numbers to something a bit more human scaled
population = [np.nansum(pop_density/10000*wmap) for wmap in weight_maps_oil]

df_oil = pd.DataFrame()
df_oil['name'] = np.ravel([names.tolist() for date in df_gfs_files.date])
df_oil['name_code'] = df_oil.name.replace(to_replace={name:i for i,name in enumerate(names)})
df_oil['date'] = np.ravel([[date] * len(names) for date in df_gfs_files.date])  
df_oil['capacity'] = np.ravel([capacities for date in df_gfs_files.date])  
df_oil['population'] =  np.ravel([population for date in df_gfs_files.date])  

df_oil.head(10)


# Before diving more into the weather data, it's maybe useful to see if there is a bigger variation of temperature/precipitation/etc along the year or between power plants.
# So I'll calculate the average mean variance in each image (spatial variation) and the variance of the mean of the images (temporal variation), and see what is bigger:

# In[ ]:


def spatial_variance(col):
    means = [np.nanmean(im) for im in df_gfs_files[col]]
    return np.std(means)

def temporal_variance(col):
    std = [np.nanstd(im) for im in df_gfs_files[col]]
    return np.mean(std)

for col in df_gfs_files.columns[2:]:
    print(col)
    print('Mean spatial std',spatial_variance(col))
    print('Mean temporal std',temporal_variance(col),'\n')


# Seems there isn't a very big difference, so we could probably adopt one value per day OR per power plant, but since we have the full data, I'm going ahead and using it.

# In[ ]:


def daily_mean_per_power_plant(im):
    return [np.nanmean(im*wmap) for wmap in weight_maps_oil]

gfs_names =  ['temperature','humidity','u_wind','v_wind','precipitable']
for col_name,col in zip(gfs_names,df_gfs_files.columns[2:]):
    df_oil[col_name]= np.ravel([daily_mean_per_power_plant(im) for im in df_gfs_files[col]])
    
df_oil.head(10)


# Now, I'll add the NO$_2$ associated with every power plant for all the days we have available, converting the units from mol/m$^2$ to mass using NO$_2$ molecular mass (46.006 g/mol) and the spatial resolution of the gfs images:
#  
# mass NO$_2$ in tonnes = $\sum_{pixels}$ sentinel image [mol/m]$^2$ $\times$ 0.0046[kg/mol] $\times$ 7000^2 [m^2/pixel] /1000
#  
# *(Note: I am a bit in doubt about the spatial resolution of these images. Looking at the header, seems that both sentinel and gfs images have the same pixels scale, which is strange because they have different native resolutions. Perhaps this was due to the extraction method of these images?... I'm going to use the 7x7 km cited in the sentinel data set, but may not be the correct value.)*

# In[ ]:


def daily_mean_NO2_per_pp(im):
    im_ton = im * 0.0046 * 7000**2 / 1000
    return [np.nansum(im_ton*wmap) for wmap in weight_maps_oil]

df_oil['NO2'] = np.ravel([daily_mean_NO2_per_pp(im) for im in df_s5p_files.NO2_column_number_density])
df_oil['cloud_frac']= np.ravel([daily_mean_per_power_plant(im) for im in df_s5p_files.cloud_fraction])
        
df_oil.head(10)


# And now we can have a look at the distribution of NO$_2$ associated with each power plant in the entire year.

# In[ ]:


fig, ax = plt.subplots(1,1,figsize=(16,5))
for i in df_oil.name_code.unique():
    sns.distplot(df_oil[df_oil.name_code==i].NO2,label=df_oil[df_oil.name_code==i].iloc[0])
    
plt.legend()


# We can have quick idea of the values of EF for each power plant, by looking at the total average.
# 
# ** NOTE: I'll also use this to have some idea of the priors for the model, but this is a wrong way of setting priors, since I'm looking at the data I'm going to fit to set the priors of the fit. The correct way would be to get this information from another independent source, or especialist about the subject (which I'm not :) ).**

# In[ ]:


df_oil.groupby('name').NO2.mean()/df_oil.groupby('name').capacity.mean()


# There's something quite suspicious about the Vieques EPP power plant (the one in the smaller island west of Puerto Rico. Since the electric capacity is much lower than the rest, the global value of the EF is actually quite higher, but seems odd that the difference is so extreme. There might be something wrong with the data of this power plant.)

# ### 4.2. Model
# 
# So, the naive model we're working with is:
# 
# > NO$_2$ = EF $\times$ energy_created + noise
# 
# And I've restricted myself to look at power plants primary fueled by oil, because the the amount of NO$_2$ generated per megawatt of power plants with different primary fuels is likely to be very different, since the way electricity is produced is very different.
# 
# However, even within the oil power plants, there might be quite big differences on how they operate (due to year of construction, owner, etc). So we expect the NO$_2$ per energy produced within these power plants to be similar, but not exactly the same. And this is where a [hierarchical bayesian model](https://twiecki.io/blog/2014/03/17/bayesian-glms-3/) could potentially be useful (I also took a lot from this very nice [post](https://towardsdatascience.com/hands-on-bayesian-statistics-with-python-pymc3-arviz-499db9a59501)) .
# 
# 
# The model will be:
# 
# >NO$_2$[power plant i] = EF[power plant i] $\times$ capacity[power plant i]
# 
# where the NO$_2$[power plant i] is the NO$_2$ associated with power plant number i, EF[power plant i] is the emission efficiency for that power plant, and capacity[power plant i] the energy generation capacity.
# 
# Now the bayes part comes from saying:
# 
# >EF[power plant i] $\propto$ $Normal$($\mu_{pp}$,$\sigma_{pp}$) 
# 
# meaning (more or less) that we cannot really measure the real value of the EF for that power plant (which would be $\mu_{pp_i}$ in this case), because the world is noisy, so what we measure from the data is *a* value of EF, that is related to $\mu_{pp_i}$ (I choose by a Normal distribution, given the amount of data we have >20 points, hopefully the [CLT](https://en.wikipedia.org/wiki/Central_limit_theorem) justifies this).
# 
# The hierarchy part comes from saying that all these $\mu_{pp_i}$ should be related, since they all concern oil power plants, so:
# 
# $\mu_{pp}$ $\propto$ $Normal$($\mu_{oil}$,$\sigma_{oil}$) 
# 
# $\sigma_{pp}$ $\propto$ HalfNormal($\beta_{oil}$) 
# 
# Neither $\sigma_{pp_i}$ or $\mu_{pp_i}$ can be negative, so I'm choosing a half normal for the first, but still allow $\mu_{pp}$ to be negative (so that the mean can be different from zero). There's might be other better choices of priors, such as the Student-t or the Cauchy distribution that would perhaps deal better with outliers, but this is sort of a base model.
# 
# I'll use pymc3 to model this.

# In[ ]:


import pymc3 as pm


# In[ ]:


with pm.Model() as hierarchical_model:
    
    # Hyperpriors
    mu_pp = pm.Normal('mu_pp', mu=0.10, sd=0.50)
    sigma_pp = pm.HalfNormal('sigma_pp', sd=0.25)
    
    # Emission Factor for each power plant, distributed around group mean mu_pp
    EF = pm.Normal('EF_oil', mu=mu_pp, sd=sigma_pp, shape=df_oil.name.nunique())
    
    # Model error. To deal with other things we still did not yet include in the model, such as population and weather
    noise = pm.HalfNormal('noise', sd=10)
    
    # Expected value of NO2 for the very naive model
    NO2 = EF[df_oil.name_code.values] * df_oil.capacity.values
    
    # Likelihood, where the mcmc magic happens
    y_like = pm.Normal('NO2', mu=NO2, sd=noise, observed=df_oil.NO2.values)


# In[ ]:


with hierarchical_model:
    hierarchical_trace = pm.sample(1000,tune=500,chains=5)


# Let's see how the chains look like

# In[ ]:


pm.traceplot(hierarchical_trace,var_names=['mu_pp','sigma_pp','EF_oil','noise']);


# And here I'm checking the results against priors (our initial, not particularly clever since I'm not a specialist, hypothesis for these parameters).

# In[ ]:


fig, ax = plt.subplots(1,3,figsize=(20,3))

ax[0].set_title('$\mu_{pp}$')
ax[0].plot(np.linspace(-1,4),scipy.stats.norm.pdf(np.linspace(-1,4),loc=0.1,scale=0.5),label='Prior')
sns.distplot(hierarchical_trace['mu_pp'],ax=ax[0],label='Posteriori')


ax[1].set_title('$\sigma_{pp}$')
ax[1].plot(np.linspace(0,4),scipy.stats.halfnorm.pdf(np.linspace(0,4),scale=0.25),label='Prior')
sns.distplot(hierarchical_trace['sigma_pp'],ax=ax[1],label='Posteriori')

ax[2].set_title('"noise"')
ax[2].plot(np.linspace(0,50),scipy.stats.halfnorm.pdf(np.linspace(0,50),scale=10),label='Prior')
sns.distplot(hierarchical_trace['noise'],ax=ax[2],label='Posteriori')

ax[2].legend()


# The posteriori distributions (in orange), which is the model "informed" by the data, seems reasoably different from our priors (in blue), which means that the data is giving us new insights in these quantities.

# And below are the posterioris for the EF of each of the power plants.

# In[ ]:


fig, ax = plt.subplots(1,6,figsize=(20,4))

for i in range(6):
    ax[i].set_title('EF '+str(np.array(names)[i]))
    sns.distplot(hierarchical_trace['EF_oil'].T[i],ax=ax[i])  
    mean_EF = np.nanmean(hierarchical_trace['EF_oil'].T[i])
    ax[i].axvline(mean_EF,color='C1')
    ax[i].annotate('EF: '+str(mean_EF.round(3)),xy=(0.1,0.9),xycoords='axes fraction')


# ### 4.2 Adding the other variables

# In[ ]:


import theano as tt


# In[ ]:


with pm.Model() as hierarchical_model_with_pop:
    
    # Hyperpriors
    mu_pp = pm.Normal('mu_pp', mu=0.30, sd=0.50)
    sigma_pp = pm.HalfNormal('sigma_pp', sd=0.25)
    
    # Emission Factor for each power plant, distributed around group mean mu_pp
    EF = pm.Normal('EF_oil', mu=mu_pp, sd=sigma_pp, shape=df_oil.name.nunique())
    
    # Model error.
    noise = pm.HalfNormal('noise', sd=10)
    
    # Add the other variables
    K = pm.Normal('K',mu=0, sd=1, shape=7)
    other_vars = tt.tensor.sum(K*df_oil[['population', 'temperature','humidity', 'u_wind', 'v_wind', 'precipitable', 'cloud_frac']],axis=1)
    
    # Expected value of NO2 for the very naive model
    NO2 = EF[df_oil.name_code.values] * df_oil.capacity.values + other_vars
    
    # Likelihood, where the mcmc magic happens
    y_like = pm.Normal('NO2', mu=NO2, sd=noise, observed=df_oil.NO2.values)   


# In[ ]:


with hierarchical_model_with_pop:
    hierarchical_trace_with_pop = pm.sample(1000,tune=1000,chains=5)


# In[ ]:


pm.traceplot(hierarchical_trace_with_pop,var_names=['mu_pp','sigma_pp','EF_oil','K','noise']);


# Check priors and posterioris again

# In[ ]:


fig, ax = plt.subplots(1,3,figsize=(20,3))

ax[0].set_title('$\mu_{pp}$')
ax[0].plot(np.linspace(0,4),scipy.stats.truncnorm.pdf(np.linspace(0,4),0,np.inf,loc=0.1,scale=0.5),label='Prior')
sns.distplot(hierarchical_trace['mu_pp'],ax=ax[0],label='Posteriori')
sns.distplot(hierarchical_trace_with_pop['mu_pp'],ax=ax[0],label='Posteriori (with K$_{pop}$)')


ax[1].set_title('$\sigma_{pp}$')
ax[1].plot(np.linspace(0,4),scipy.stats.halfnorm.pdf(np.linspace(0,4),scale=0.25),label='Prior')
sns.distplot(hierarchical_trace['sigma_pp'],ax=ax[1],label='Posteriori')
sns.distplot(hierarchical_trace_with_pop['sigma_pp'],ax=ax[1],label='Posteriori (with other vars$)')

ax[2].set_title('"noise"')
ax[2].plot(np.linspace(0,50),scipy.stats.halfnorm.pdf(np.linspace(0,50),scale=10),label='Prior')
sns.distplot(hierarchical_trace['noise'],ax=ax[2],label='Posteriori')
sns.distplot(hierarchical_trace_with_pop['noise'],ax=ax[2],label='Posteriori (with other vars$)')

ax[2].legend();


# Althought the mean value of global EF for oil power plant ($\mu_{pp}$) doesn't dramatically change when adding the other variables, the 'noise' is quite lower, which might be a good sign.

# We can also have a look at the coeficients associated with population, temperature, etc in this linear model:

# In[ ]:


fig, ax = plt.subplots(1,7,figsize=(20,3))

other_var_names = ['population', 'temperature','humidity', 'u_wind', 'v_wind', 'precipitable', 'cloud_frac']

for i in range(7):
    ax[i].set_title('K '+str(np.array(other_var_names)[i]))
    sns.distplot(hierarchical_trace_with_pop['K'].T[i],ax=ax[i])  
    mean_K = np.nanmean(hierarchical_trace_with_pop['K'].T[i])
    ax[i].axvline(mean_K,color='C1')
    ax[i].annotate('K: '+str(mean_K.round(4)),xy=(0.05,0.9),xycoords='axes fraction')


# Although the absolute value of the Ks depend on the magnitude of the several variables, so it might be hard to intrepret since I didn't normalise them, we can see that population seems to be adding to the NO$_2$ (positive coefficient), so as temperature, the v component of the wind and the precipitable water in the atmosphere.

# And finally the yearly avaerage EF for each of the Oil power plants, which is what I was trying to calculate in this analyis.

# In[ ]:


fig, ax = plt.subplots(1,6,figsize=(20,4))

for i in range(6):
    ax[i].set_title('K '+str(np.array(names)[i]))
    sns.distplot(hierarchical_trace_with_pop['EF_oil'].T[i],ax=ax[i])  
    mean_EF = np.nanmean(hierarchical_trace_with_pop['EF_oil'].T[i])
    ax[i].axvline(mean_EF,color='C1')
    ax[i].annotate('EF: '+str(mean_EF.round(4)),xy=(0.05,0.9),xycoords='axes fraction')


# ## 5. Conclusion
# 
# In this analysis, I've focused on calculating the yearly average EF for each of the power plants that run primaraly on oil in Puerto Rico.
# 
# The first step was to attribute the NO$_2$ measured in satelite images to a power plant, based on their distance to each pixel. This is something that can be improved (see above).
# 
# Then I used a hierarquical bayesian model to calculate the EF. The advantage of this method is that i) one could incorporate smart priors, for example the previously measured EF through the more manual methods or some EF measured in other regions, to improve the results, ii) it gives more detailled information than just calculating an average EF for oil power plants, but still combines the information of all power plants to calculate the individual EFs, which makes the method more robust of noisy data for one particular power plant.
