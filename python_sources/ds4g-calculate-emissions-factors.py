#!/usr/bin/env python
# coding: utf-8

# ### **DS4G: Environmental Insights Explorer**
# #### Exploring alternatives for emissions factor calculations  
# 
# Allison Smith

# **Objective**: Develop a methodology to calculate an average annual historical emissions factor for the sub-national region.  
# 
# **Bonus Objective 1**: Smaller time slices of the average historical emissions factors, such as one per month for the 12-month period
# 
# **Bonus Objective 2**: Develop methodologies for calculating marginal emissions factors for the sub-national region.

# ### Abstract
# 
# The code in the submission will produce an annual emissions factor, monthly emissions factors, and marginal emissions factors for nitrogen dioxide from power plants in Puerto Rico using data from satellite based sensors.  The methodology combines geochemistry, grid analysis, and predictive modeling.  There are also sections discussing sources of nitrogen dioxide, assumptions of the methods, validation that is needed, and extensions to additional sub-national regions.

# ### Table of Contents
# 
# Sources of $NO_2$ emissions
# 
# Grid Cell Methodology
# 
# Simplified Emissions Factor
# 
# Convert .tif to NetCDF (.nc) format
# 
# Regrid Data
# 
# Calculate the Annual Emissions Factor
# 
# Calculate the Monthly Emissions Factors
# 
# Calculate the Marginal Emission Factors
# 
# Conclusions
# 
# Validation 
# 
# Extension to Other Sub-national Regions
# 
# Additional Thoughts
# 

# ### Sources of $NO_2$ emissions
# 
# Primary sources of $NO_{2}$ emissions are from power plants, motor vehicles, and silage<sup>1</sup>.  The $NO_{2}$ data from the sensors on the Sentinel 5P OFFL satellite are a combination of emissions from all of these sources. In mathematical terms:  
# 
# $NO_{2_{Sentinel}} = NO_{2_{powerplants}} + NO_{2_{vehicles}} + NO_{2_{silage}}$ 
# 
# The objective is to calculate an emissions factor for power plants.  In order to achieve this objective, the fraction of emissions measured by the sensor must be attributed to power plants. Power plant emissions are difficult to attribute because of the variability in the characteristics of power plants, which use different types of fuels, have different pollution scrubbers, and have fluctuating emissions depending on demand. 
# 
# $NO_{2}$ emissions from motor vehicles are less impacted by the emissions from any individual vehicle. Therefore, the collective emissions across many vehicles can be generalized as long as there are no contributions from power plants or silage. 
# 
# $NO_{2_{vehicles}} = NO_{2_{Sentinel}}$ 
# 
# where $NO_{2_{powerplants}} = 0$ and $NO_{2_{silage}} = 0$
# 
# Silage describes the process of fermenting green foliage crops to be used for animal feed.  $NO_{2}$ emissions from silage occur in agricultural areas and have a seasonal component depending on the harvesting patterns of a geographic area.  For the purposes of this analysis, I will make the assumption that grid cells with power plants do not have large agricultural operations.    
# 
# Therefore, $NO_2$ from powerplants would be calculated as: 
# 
# $NO_{2_{powerplants}} = NO_{2_{Sentinel}} - NO_{2_{vehicles}}$
# 
# where $NO_{2_{silage}} = 0$
# 
# Reference:   
# <sup>1</sup>https://toxtown.nlm.nih.gov/chemicals-and-contaminants/nitrogen-oxides

# ### Grid cell methodology
# 
# Satellite sensors and data assimilation products provide data on grids.  I will regrid the data to 0.01 degree latitude by 0.01 degree longitude grid cells for Puerto Rico which is approximately 1 km x 1 km.  Grid cells will be divided based on primary sources of $NO_2$.  Grid cells with agriculture will be excluded from analyses in order exclude the potential effects of $NO_2$ from silage which is outside the scope of the objectives.  Grid cells without power plants or agriculture will be used to build a predictive model for $NO_2$ emissions from motor vehicles in a grid cell.  Then the predictive model will be used to predict the $NO_2$ emissions from motor vehicles in grid cells with power plants.  The $NO_2$ emissions from motor vehicles can then be subtracted from the total $NO_2$ emissions measured by the satellite to get the $NO_2$ emissions produced by the power plant.  These $NO_2$ emissions can then be used to calculate the emissions factor.  
# 
# There are many assumptions with this method.  
# 
# There is an assumption that all $NO_2$ emissions in grid cells without power plants and agriculture comes from motor vehicles. 
# 
# There is an assumption that there is little advection of $NO_2$ from the grid cells near power plants.  I filter the data for the days with the lowest wind speeds but there is still advection. 
# 
# There is an assumption that the residence time for $NO_2$ is approximately 1 day. However, the residence time of $NO_2$ varies over time depending on local environmental conditions.  If the residence time is less than 1 day, the emissions will be under estimated. Whereas, if the residence time is more than 1 day , the emissions will be over estimated. 

# ### Simplified Emissions Factor
# 
# I will use the simplified emissions factor that was presented in the overview for the Kaggle challenge because I do not have data for emissions reduction technology for the power plants in Puerto Rico. 
# 
# Emissions = Activity_rate * Emissions_factor
# 
# Rearranging: 
# 
# Emissions_factor = Emissions/Activity_rate
# 
# $NO_2$ Emissions Factor:
# 
# Emissions_factor = $NO_{2_{powerplants}}$/estimated_generation_gwh
# 
# where the $NO_{2_{powerplants}}$ is calculated using the grid cell methodology and the estimated_generation_gwh is from the Global Database of Power Plants. 

# ### Convert .tif Files to NetCDF (.nc)
# Shell script for converting the s5p_no2 data in the starter pack from multiple .tif files into a single .nc file. I converted all of the data sets because I find the .nc format to be more intuitive and easier to use for analyses.  

# In[ ]:


##Shell Script
##requires gdal, nco, cdo to run 
##change directory
# cd ../data/starter_pack/s5p_no2/

# filenamelist=`ls tif/*.tif`
# for tif_file in $filenamelist
# do

#     echo $tif_file
#     #tif_file="tif/s5p_no2_20180701T161259_20180707T175356.tif"

#     #set names
#     filename=${tif_file%.tif}
#     nc_file=nc/${filename#tif/}.nc

#     #convert tif to netCDF
#     gdal_translate -of NETCDF -co "FORMAT=NC4" ${tif_file} convert.nc

#     #rename th variables in the netcdf files
#     ncrename -v Band1,NO2_column_number_density convert.nc
#     ncrename -v Band2,tropospheric_NO2_column_number_density convert.nc
#     ncrename -v Band3,stratospheric_NO2_column_number_density convert.nc
#     ncrename -v Band4,NO2_slant_column_number_density convert.nc
#     ncrename -v Band5,tropopause_pressure convert.nc
#     ncrename -v Band6,absorbing_aerosol_index convert.nc
#     ncrename -v Band7,cloud_fraction convert.nc
#     ncrename -v Band8,sensor_altitude convert.nc
#     ncrename -v Band9,sensor_azimuth_angle convert.nc
#     ncrename -v Band10,sensor_zenith_angle convert.nc
#     ncrename -v Band11,solar_azimuth_angle convert.nc
#     ncrename -v Band12,solar_zenith_angle convert.nc

#     #extract time from the file name and add as a dimension to the netCDF file
#     timestrT="$(cut -d'_' -f3 <<<${filename})"
#     timestr="${timestrT//T}"
#     unix="$(date -j -u -f "%Y%m%d%H%M%S" "${timestr}" +"%s")"
#     # printf "/***.nco ncap2 script***/
#     # defdim(\"time\",1);
#     # time[time]="$unix";
#     # time@long_name=\"Time\";
#     # time@units=\"seconds since 1970-01-01 00:00:00\";
#     # time@standard_name=\"time\";
#     # time@axis=\"T\";
#     # time@coordinate_defines=\"point\";
#     # time@calendar=\"standard\";
#     # /***********/" > time.nco

#     ncap2 -Oh -s 'defdim("time",1);time[time]=1561830483;time@long_name="Time";time@units="seconds since 1970-01-01 00:00:00";time@standard_name="time";time@axis="T";time@coordinate_defines="point";time@calendar="standard";' convert.nc convert1.nc
#     date_string="$(date -u -r ${unix} +'%Y-%m-%d,%H:%M:%S')"
#     cdo settaxis,"${date_string}" convert1.nc convert2.nc

#     #save final version of netcdf file 
#     mv convert2.nc ${nc_file}

#     #clean up files
#     rm convert.nc
#     rm convert1.nc
#     #rm time.nco 

# done

# cdo mergetime nc/*.nc no2_1year.nc


# ### Python packages

# In[ ]:


import numpy as np
import pandas as pd
import xarray as xr
#import xesmf as xe
import json
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns
import json
import datetime as dt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ### Regrid data
# 
# **The files that result from running the regridding code in this section were uploaded to Kaggle to be used to calculate the emissions factor.**  They are annual_ds4g_emissions.nc and monthly_ds4g_emissions.nc.  
# 
# 
# The code in this section requires xesmf and additional package dependencies to be installed.  The NetCDF files that were generated by the shell script were also too large to upload.   

# The data sets have different latitude, longitude, and time intervals. The data need to be converted to the same intervals, which I will refer to as grids, for analysis. I regridded the data using the xesmf package in python which makes regridding much easier.

# In[ ]:


#Reusable function for doing the regridding:
# def make_regridder(ds, ds_base, variable, algorithm='bilinear'):   
#     if 'latitude' in ds[variable].dims:
#        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'}).set_coords(['lon', 'lat'])
#     ds_regrid = xr.Dataset({'lat': (['lat'], np.arange(np.floor(ds_base['lat'].min().values*10)/10, np.ceil(ds_base['lat'].max().values*10)/10, 0.01)),
#                      'lon': (['lon'], np.arange(np.floor(ds_base['lon'].min().values*10)/10, np.ceil(ds_base['lon'].max().values*10)/10, 0.01)),
#                     }
#                    )

#     regridder = xe.Regridder(ds, ds_regrid, algorithm)
#     regridder.clean_weight_file()
#     return regridder


# #### Create a base grid

# In[ ]:


#Base grid is based on the s5p_no2 data, i.e. all other data sets will be regrid to the no2 grid
# ds_s5p = xr.open_dataset('../data/starter_pack/s5p_no2/no2_1year.nc')
# ds_no2_clouds = ds_s5p[['NO2_column_number_density', 'cloud_fraction']]
# no2_regridder = make_regridder(ds_no2_clouds, ds_no2_clouds, 'NO2_column_number_density')
# ds_base_regrid = no2_regridder(ds_no2_clouds)
# ds_base_regrid = ds_base_regrid.where(ds_base_regrid['NO2_column_number_density']!=0.)


# In[ ]:


#Plot the NO_2 data for a single time point
# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))
# ds_base_regrid.NO2_column_number_density.isel(time=0).plot(ax=ax, transform=ccrs.PlateCarree());
# ax.coastlines()
# ax.set_extent([-67.5, -65, 17.5, 19])
# ax.set_aspect("equal")


# #### Add land mask to the base grid
# In the plot above, there are $NO_2$ measurements both over land and sea.  The sources of $NO_2$ most critical for these analyses are based on land so a high resolution land mask will be useful for doing analyses and creating plots. I typically use a very high resolution SST file (1 km) to create land masks. A freely available data set can be accessed via the link below. Google earth engine only has the GHRSST 4 km data. The data need to be downloaded from NOAA directly to get the 1 km data.

# In[ ]:


#Download Super High Resolution GHRSST SST file (0.01 degree grid)
#https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplG1SST.nc?SST[(2017-09-13T00:00:00Z):1:(2017-09-13T00:00:00Z)][(17.005):1:(19.005)][(-69.995):1:(-64.005)],mask[(2017-09-13T00:00:00Z):1:(2017-09-13T00:00:00Z)][(17.005):1:(19.005)][(-69.995):1:(-64.005)],analysis_error[(2017-09-13T00:00:00Z):1:(2017-09-13T00:00:00Z)][(17.005):1:(19.005)][(-69.995):1:(-64.005)]


# In[ ]:


#Read in SST data, the file is in the Kaggle folder input/ds4g-emissions-nc
# ds_sea = xr.open_dataset('../input/ds4g-emissions-nc/jplG1SST_e435_8209_9395.nc')


# In[ ]:


#Plot SST to view the grid
# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))
# ds_sea.SST.isel(time=0).plot(ax=ax, transform=ccrs.PlateCarree());
# ax.coastlines()
# ax.set_extent([-67.5, -65, 17.5, 19])
# ax.set_aspect("equal")


# In[ ]:


#Regrid SST data to the same grid as the NO2 data
# sea_regridder = make_regridder(ds_sea, ds_base_regrid, 'SST')
# ds_sea_regrid = sea_regridder(ds_sea)
# ds_sea_regrid = ds_sea_regrid.where(ds_sea_regrid['SST']!=0.)


# In[ ]:


#Plot the regridded data
# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))
# ds_sea_regrid.SST.isel(time=0).plot(ax=ax, transform=ccrs.PlateCarree());
# ax.set_extent([-67.5, -65, 17.5, 19])
# ax.set_aspect("equal")


# In[ ]:


#Create a land mask and add it to the base data set
# land_ones = ds_sea_regrid.SST.isel(time=0).fillna(1)
# land_mask = land_ones.where(land_ones ==1.)
# land_mask = land_mask.where(land_mask.lat<18.5)
# land_mask = land_mask.drop('time')
# ds_base_regrid.coords['land_mask'] = land_mask


# In[ ]:


#Plot NO2 for just the land
# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))
# ds_base_regrid['NO2_column_number_density'].isel(time=0).where(ds_base_regrid.land_mask == 1).plot(ax=ax, transform=ccrs.PlateCarree())
# ax.set_extent([-67.5, -65, 17.5, 19])
# ax.set_aspect("equal")


# #### Compute daily averages for s5p_no2 data
# The s5p_no2 data are available on nearly daily intervals. Occasionally, multiple data points are recorded on the same day.  The next step is to standardize the data to a daily time interval. 

# In[ ]:


# ds_base = ds_base_regrid.resample(time='1D').mean()


# In[ ]:


#Note the changes to the timestamp when the data are plotted
# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))
# ds_base['NO2_column_number_density'].isel(time=0).where(ds_base.land_mask == 1).plot(ax=ax, transform=ccrs.PlateCarree())
# ax.set_extent([-67.5, -65, 17.5, 19])
# ax.set_aspect("equal")


# #### Add weather data to the base grid
# There were two data sets in the starter pack that included weather data - the gfs data set and the gldas data set.  The gfs data set is based on forecasts and the gldas data set is based on recorded data. There is almost complete overlap in the variables available in each of the data sets. I decided to mostly use the gldas data set because recorded data is better than forecasts for historical assessments.  The gldas data set also has more variables available. 

# In[ ]:


# ds_gldas = xr.open_dataset('../data/starter_pack/gldas/gldas_1year.nc')
# ds_gldas = ds_gldas.drop('crs')
# gldas_regridder = make_regridder(ds_gldas, ds_base_regrid, 'Tair_f_inst',  'nearest_s2d')
# ds_gldas_regrid = gldas_regridder(ds_gldas)
# ds_gldas_regrid = ds_gldas_regrid.where(ds_gldas_regrid['Tair_f_inst']!=0.)
# ds_gldas_regrid.coords['land_mask'] = land_mask


# The gldas data are only available for land and the original grid size is fairly large and does not cover many of the coastal sites where power plants are located in Puerto Rico. I filled the coastal areas with the nearest neighbor pixel.  

# In[ ]:


# ds_gldas_regrid_fill = ds_gldas_regrid.ffill(dim='lat')
# ds_gldas_regrid_fill = ds_gldas_regrid_fill.bfill(dim='lat')
# ds_gldas_regrid_fill = ds_gldas_regrid_fill.ffill(dim='lon')
# ds_gldas_regrid_fill = ds_gldas_regrid_fill.bfill(dim='lon')


# In[ ]:


# #The gldas data without the coastal pixels filled
# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))
# ds_gldas_regrid['Tair_f_inst'].isel(time=0).where(land_mask == 1).plot(ax=ax, transform=ccrs.PlateCarree())
# ax.set_extent([-67.5, -65, 17.5, 19])
# ax.set_aspect("equal")


# In[ ]:


# #The gldas data with the coastal pixels filled
# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))
# ds_gldas_regrid_fill['Tair_f_inst'].isel(time=0).where(land_mask == 1).plot(ax=ax, transform=ccrs.PlateCarree())
# ax.set_extent([-67.5, -65, 17.5, 19])
# ax.set_aspect("equal")


# The gldas data are available at 3 hour intervals. Daily mean, max, and min have the potential to be useful depending on the weather variable being measured. 

# In[ ]:


# ds_gldas_daily_mean = ds_gldas_regrid_fill.resample(time='1D').mean()
# ds_gldas_daily_max = ds_gldas_regrid_fill.resample(time='1D').max()
# ds_gldas_daily_min = ds_gldas_regrid_fill.resample(time='1D').min()


# In[ ]:


# ds_base['gldas_wind_mean'] = ds_gldas_daily_mean['Wind_f_inst']
# ds_base['gldas_airT_mean'] = ds_gldas_daily_mean['Tair_f_inst']
# ds_base['gldas_airT_max'] = ds_gldas_daily_max['Tair_f_inst']
# ds_base['gldas_airT_min'] = ds_gldas_daily_min['Tair_f_inst']
# ds_base['gldas_lwdown_mean'] = ds_gldas_daily_mean['LWdown_f_tavg']
# ds_base['gldas_pres_mean'] = ds_gldas_daily_mean['Psurf_f_inst']
# ds_base['gldas_humidity_mean'] = ds_gldas_daily_mean['Qair_f_inst']
# ds_base['gldas_heatflux_mean'] = ds_gldas_daily_mean['Qg_tavg']
# ds_base['gldas_rain_max'] = ds_gldas_daily_max['Rainf_f_tavg']
# ds_base['gldas_SWdown_max'] = ds_gldas_daily_max['SWdown_f_tavg']


# #### Add GFS wind data to the base grid
# I used GFS for wind speed data because GFS had coastal coverage without filling with the nearest pixel. Wind is particularly important for determining emissions.  Advection of $NO_2$ away from point source grid cells occurs on windy days.  

# In[ ]:


# ds_gfs = xr.open_dataset('../data/starter_pack/gfs/gfs_1year.nc')
# ds_gfs = ds_gfs.drop('crs')
# gfs_regridder = make_regridder(ds_gfs, ds_base_regrid, 'temperature_2m_above_ground')
# ds_gfs_regrid = gfs_regridder(ds_gfs)
# ds_gfs_regrid = ds_gfs_regrid.where(ds_gfs_regrid['temperature_2m_above_ground']!=0.)
# ds_gfs_regrid.coords['land_mask'] = land_mask
# ds_gfs_regrid['wind_speed'] = np.sqrt(np.square(ds_gfs_regrid.u_component_of_wind_10m_above_ground) + np.square(ds_gfs_regrid.v_component_of_wind_10m_above_ground))


# In[ ]:


# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))
# ds_gfs_regrid['wind_speed'].isel(time=6).where(ds_gfs_regrid.land_mask == 1).plot(ax=ax, transform=ccrs.PlateCarree())
# ax.set_extent([-67.5, -65, 17.5, 19])
# ax.set_aspect("equal")


# In[ ]:


# ds_gfs_daily_mean = ds_gfs_regrid.resample(time='1D').mean()
# ds_base['gfs_wind_mean'] = ds_gfs_daily_mean['wind_speed']


# #### Add night time lights data to the base grid
# Additional data are needed to calculate the $NO_2$ emissions from motor vehicles.  Ideally, a gridded data set with the number of gas stations per grid cell would be available in Google Earth Engine. This data set is not available (yet). The options that I did find in Google Earth Engine that could be proxies for motor vehicles are population density and nighttime lights. The 
# population density data product available on Google Earth Engine is the GPWv411: Population Density (Gridded Population of the World Version 4.11). The most recent version of night time lights is the VIIRS Nighttime Day/Night Band Composites Version 1.
# 
# I decided to use the night time lights because the population density are modeled estimates based on censuses and the night time lights are data regularly captured with a sensor on a satellite. Data captured regularly via a satellite platform tend to be more reliable and up-to-date.   
# 
# https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMCFG
# 

# In[ ]:


##Data were downloaded using the Google Earth Engine Interface.  
# // Create a geometry representing an export region.
# var geometry = ee.Geometry.Rectangle([-67.4, 17.9, -65.1, 18.6])

# // Load an image
# var getimagecollection = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
#       .filterDate('2018-07-01', '2019-07-01')
#       .filterBounds(geometry);

# var imageselect = getimagecollection.select('avg_rad').first();

# // var getimage = ee.Image(imageselect).first();


# // Export the image, specifying scale and region.
# Export.image.toDrive({
# image: imageselect,
# description: 'ds4g_nighttime_lights2',
# scale: 1000,
# region: geometry,
# });


# In[ ]:


# ds_nightlights = xr.open_dataset('../input/ds4g-emissions-nc/VIIRS_nighttime_lights.nc')
# ds_nightlights2 = ds_nightlights.drop('crs')


# In[ ]:


# nl_regridder = make_regridder(ds_nightlights2, ds_base_regrid, 'avg_rad')

# ds_nl_regrid = nl_regridder(ds_nightlights2)
# ds_nl_regrid = ds_nl_regrid.where(ds_nl_regrid['avg_rad']!=0.)
# ds_nl_regrid.coords['land_mask'] = land_mask


# In[ ]:


# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))
# ds_nl_regrid['avg_rad'].where(ds_nl_regrid.land_mask == 1).plot(ax=ax, transform=ccrs.PlateCarree())
# ax.set_extent([-67.5, -65, 17.5, 19])
# ax.set_aspect("equal")


# In[ ]:


# ds_base['night_avg_rad'] = ds_nl_regrid['avg_rad']


# #### Add land cover data to the base grid
# Additional data are needed to exclude the $NO_2$ emissions from silage.  There are some land cover data sets available on Google Earth Engine. I used the GFSAD1000: Cropland Extent 1km Multi-Study Crop Mask, Global Food-Support Analysis Data becuase it was global and had a high resolution. A global data set is key to applying this method to other sub-national regions.  
# 
# https://developers.google.com/earth-engine/datasets/catalog/USGS_GFSAD1000_V1

# In[ ]:


##Data were downloaded using the Google Earth Engine Interface.  
# // Create a geometry representing an export region.
# var geometry = ee.Geometry.Rectangle([-67.4, 17.9, -65.1, 18.6])

# // Load an image
# var getimage = ee.Image("USGS/GFSAD1000_V1")

# var imageselect = getimage.select('landcover');

# // var getimage = ee.Image(imageselect).first();


# // Export the image, specifying scale and region.
# Export.image.toDrive({
# image: imageselect,
# description: 'ds4g_landcover',
# scale: 1000,
# region: geometry,
# });


# In[ ]:


# ds_landcover = xr.open_dataset('../input/ds4g-emissions-nc/GFSAD1000_landcover.nc')
# ds_landcover = ds_landcover.drop('crs')
# land_regridder = make_regridder(ds_landcover, ds_base_regrid, 'landcover_category',  'nearest_s2d')
# ds_land_regrid = land_regridder(ds_landcover)
# ds_land_regrid.coords['land_mask'] = land_mask


# In[ ]:


# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))
# ds_land_regrid['landcover_category'].where(ds_land_regrid.land_mask == 1).plot(ax=ax, transform=ccrs.PlateCarree())
# ax.set_extent([-67.5, -65, 17.5, 19])
# ax.set_aspect("equal")


# In[ ]:


# ds_base['landcover_category'] = ds_land_regrid['landcover_category']


# #### Add masks for power plants to the base grid

# The grid cells that contain power plants need to be selected with a mask to analyze the emissions from power plants.  I also created a mask to exclude the grid cells with power plants from the analysis of the emissions from motor vehicles. 

# In[ ]:


#Read in the data 
# plants = pd.read_csv('../input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')
# plants = plants[['capacity_mw', 'estimated_generation_gwh', 'primary_fuel', '.geo']]

#Extract the latitude and longitude coordinates from a json string into separate columns
# coordinates = pd.json_normalize(plants['.geo'].apply(json.loads))['coordinates']
# plants[['longitude', 'latitude']] = pd.DataFrame(coordinates.values.tolist(), index= coordinates.index)
# plants.drop('.geo', axis=1, inplace=True)


# pd.json_normalize was missing from Kaggle, I uploaded the data with latitude and longitude as separate columns
plants = pd.read_csv('../input/ds4g-emissions-nc/gppd_120_pr_lat_lon.csv')


# In[ ]:


#Filter for plants that burn fossil fuels and generate NO2
plants_fossil = plants[plants['primary_fuel'].isin(['Oil', 'Gas', 'Coal'])].copy()
plants_fossil.reset_index(drop=True, inplace=True)
# plants_fossil['grid_lon'] = np.nan
# plants_fossil['position_lon'] = np.ones
# plants_fossil['grid_lat'] = np.nan
# plants_fossil['position_lat'] = np.ones

#Map the fossil fuel plants to the nearest grid cells in the base grid 
# lons = ds_base.lon.values
# a=0 
# for lon in plants_fossil.longitude:
#     lon_diff = abs(lon-lons) 
#     plants_fossil.at[a,'grid_lon'] = lons[np.argmin(lon_diff)]
#     plants_fossil.at[a,'position_lon'] = np.argmin(lon_diff)
#     a=a+1

# lats = ds_base.lat.values
# a=0 
# for lat in plants_fossil.latitude:
#     lat_diff = abs(lat-lats) 
#     plants_fossil.at[a,'grid_lat'] = lats[np.argmin(lat_diff)]
#     plants_fossil.at[a,'position_lat'] = np.argmin(lat_diff)
#     a=a+1


# In[ ]:


#Calculate the number of plants in each grid cell
plants_fossil['num_plants'] = 1
#plants_fossil_grid = plants_fossil[['grid_lon', 'grid_lat', 'position_lat', 'position_lon', 'num_plants']].groupby(['grid_lon', 'grid_lat', 'position_lat', 'position_lon'], as_index=False).sum()
plants_fossil_grid = pd.read_csv('../input/ds4g-emissions-nc/plants_fossil_grid.csv')


# In[ ]:


#Save data for future use
#plants_fossil.to_csv('plants_fossil.csv', index=False)


# In[ ]:


# Convert data frame into a grid that can be added as a mask in the base grid
# Also creating a mask with the positions of the latitude and longitudes in the grid to use for the marginal emissions analysis
# plants_mask = 0 * np.ones((ds_base.dims['lat'], ds_base.dims['lon'])) * np.isnan(ds_base.NO2_column_number_density.isel(time=0)) 
# position_lat_id = 0 * np.ones((ds_base.dims['lat'], ds_base.dims['lon'])) * np.isnan(ds_base.NO2_column_number_density.isel(time=0))
# position_lon_id = 0 * np.ones((ds_base.dims['lat'], ds_base.dims['lon'])) * np.isnan(ds_base.NO2_column_number_density.isel(time=0))
# plants_mask = plants_mask.drop('time')

# # Create masks for the fossil fuel power plants, 
# # The mask also includes the grid cells immediately surround the grid cell with the power plant
# for x in plants_fossil_grid.index:
#     plants_mask[(plants_fossil_grid.at[x,'position_lat']-2):(plants_fossil_grid.at[x,'position_lat']+2),(plants_fossil_grid.at[x,'position_lon']-2):(plants_fossil_grid.at[x,'position_lon']+2)]=1
#     position_lat_id[(plants_fossil_grid.at[x,'position_lat']-2):(plants_fossil_grid.at[x,'position_lat']+2),(plants_fossil_grid.at[x,'position_lon']-2):(plants_fossil_grid.at[x,'position_lon']+2)]=plants_fossil_grid.at[x,'position_lat']
#     position_lon_id[(plants_fossil_grid.at[x,'position_lat']-2):(plants_fossil_grid.at[x,'position_lat']+2),(plants_fossil_grid.at[x,'position_lon']-2):(plants_fossil_grid.at[x,'position_lon']+2)]=plants_fossil_grid.at[x,'position_lon']

# # Add the masks to the base grid array    
# plants_mask = plants_mask.where(plants_mask == 1.)
# position_lat_id = position_lat_id.where(position_lat_id >= 1.)
# position_lon_id = position_lon_id.where(position_lon_id >= 1.)
# ds_base.coords['plants_mask'] = (('lat', 'lon'), plants_mask)
# ds_base.coords['no_plants_mask'] = ds_base.plants_mask.fillna(0).where((ds_base.plants_mask != 1) & (ds_base.land_mask == 1))
# ds_base.coords['no_plants_mask']  = ds_base.no_plants_mask + 1
# ds_base.coords['position_lat_id'] = (('lat', 'lon'), position_lat_id)
# ds_base.coords['position_lat_id'] = ds_base.position_lat_id.where(ds_base.position_lat_id >= 1)
# ds_base.coords['position_lon_id'] = (('lat', 'lon'), position_lon_id)
# ds_base.coords['position_lon_id'] = ds_base.position_lon_id.where(ds_base.position_lon_id >= 1)


# In[ ]:


# Plot the grid cell areas with power plants 
# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))
# ds_base['NO2_column_number_density'].isel(time=0).where((land_mask==1) & (plants_mask==1)).plot(ax=ax, transform=ccrs.PlateCarree())
# ax.set_extent([-67.5, -65, 17.5, 19])
# ax.set_aspect("equal")


# In[ ]:


# Plot the grid cell ares without power plants
# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))
# ds_base['NO2_column_number_density'].isel(time=0).where(ds_base.no_plants_mask == 1).plot(ax=ax, transform=ccrs.PlateCarree())
# ax.set_extent([-67.5, -65, 17.5, 19])
# ax.set_aspect("equal")


# ### Calculate daily average for the year

# $NO_2$ emission concentrations are variable because $NO_2$ can be advected by wind. I filtered the data for the days with the lowest wind speeds and then calculated an annual average for the geographic area.  

# In[ ]:


# ds_base_annual = ds_base.where((ds_base.gfs_wind_mean <= 2)).mean(dim=['time'])


# In[ ]:


# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))
# ds_base_annual['NO2_column_number_density'].where((ds_base_annual.land_mask == 1) & (ds_base_annual.no_plants_mask ==1)).plot(ax=ax, transform=ccrs.PlateCarree())
# ax.set_extent([-67.5, -65, 17.5, 19])
# ax.set_aspect("equal")


# In[ ]:


#Base grid file is finished. 
#Save a copy to a NetCDF file to be used to calculate the emissions factor.
#ds_base_annual.to_netcdf('annual_ds4g_emissions.nc')


# ### Calculate daily averages for each month

# I also calculated monthly averages but the threshold filter for wind speed had to be considerably higher to get adequate geographic coverage for a month. 

# In[ ]:


# ds_base_monthly = ds_base.where((ds_base.gfs_wind_mean <= 5)).resample(time='1M').mean()


# In[ ]:


# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))
# ds_base_monthly['NO2_column_number_density'].isel(time=0).where(ds_base_monthly.land_mask == 1).plot(ax=ax, transform=ccrs.PlateCarree())
# ax.set_extent([-67.5, -65, 17.5, 19])
# ax.set_aspect("equal")


# In[ ]:


#Save a copy to a NetCDF file to be used to calculate the emissions factor for each month.
#ds_base_monthly.to_netcdf('monthly_ds4g_emissions.nc')


# ### Calculate the annual emissions factor

# In[ ]:


#Read in the data set from the saved file created using the code in the regridding section above.
ds = xr.open_dataset('../input/ds4g-emissions-nc/annual_ds4g_emissions.nc')

#Use data set created above.
#ds = ds_base_annual


# In[ ]:


print(ds)


# In[ ]:


#Create a plot of the NO2 emissions from the data set
ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))
ds.NO2_column_number_density.where(ds.land_mask==1).plot(ax=ax, transform=ccrs.PlateCarree());


# In[ ]:


#Create a plot of the gfs wind speed from the data set
ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))
ds.wind_speed_mean.where(ds.land_mask==1).plot(ax=ax, transform=ccrs.PlateCarree());


# In[ ]:


#Create a plot of the NO2 emissions for just the power plant grid cells
ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))
ds.NO2_column_number_density.where(ds.plants_mask==1).plot(ax=ax, transform=ccrs.PlateCarree());


# #### Build model to predict motor vehicle $NO_2$ emissions for each grid cell
# Use masks and landcover data to filter for grid cells that are located on land, have no power plants nearby, and do not have agricultural operations. 

# In[ ]:


ds['landcover_category'] = ds['landcover_category'].fillna(0)
ds_land = ds.where((ds.land_mask == 1) & 
                   (ds.no_plants_mask == 1) & 
                   (ds.landcover_category == 0))


# In[ ]:


ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))
ds_land.NO2_column_number_density.plot(ax=ax, transform=ccrs.PlateCarree());


# In[ ]:


ds_vehicles = ds_land
ds_vehicles = ds_vehicles.drop(['no_plants_mask','plants_mask','land_mask', 'position_lat_id', 'position_lon_id'])
df_vehicles = ds_vehicles.to_dataframe().dropna()


# In[ ]:


# Explore features
# df_vehicles.describe()
# sns.pairplot(df_vehicles)


# Extract features that are important for predicting $NO_2$ emissions in grid cells where the main source of $NO_2$ emissions is motor vehicles. 

# In[ ]:


df_vehicles = df_vehicles[['NO2_column_number_density', 'cloud_fraction', 'night_avg_rad', 'gldas_wind_mean', 'gldas_airT_max', 'gldas_pres_mean', 'gldas_lwdown_mean']].copy()
df_vehicles['arcsin_cloud_fraction'] = np.arcsin(np.sqrt(df_vehicles['cloud_fraction']))
df_vehicles['log_night_avg_rad'] = np.log(df_vehicles['night_avg_rad'])
df_vehicles.drop(['cloud_fraction', 'night_avg_rad'], axis=1, inplace=True)


# In[ ]:


# Generate a plot to make sure that none of the variables are highly correlated. 

df_vehicles2 = df_vehicles.copy()
df_vehicles2['NO2_column_number_density'] = df_vehicles['NO2_column_number_density']*(10**5) #Kaggle does not always plot very small numbers so rescaling.
sns.pairplot(df_vehicles2)


# Build an XGBoost model to predict the NO2_column_number_density for grid cells in which the primary source of $NO_2$ emissions is from motor vehicles. 

# In[ ]:


X = np.array(df_vehicles.drop(['NO2_column_number_density'], axis=1))
y = np.array(df_vehicles['NO2_column_number_density'])
y = y*(10**5) #The numbers are very small so rescaling for model training and prediction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=122)
model = XGBRegressor(learning_rate = 0.01, objective='reg:squarederror', n_estimators = 500)
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)


# In[ ]:


plt.scatter(y_test, y_pred) #Kaggle has limits on small scales, so these should be 10**-5


# Check the performance of the model. The metrics indicate that the data are not overfit during training.

# In[ ]:


print(r2_score(y_train, y_train_pred))
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test*10**-5, y_pred*10**-5))
print(mean_absolute_error(y_test*10**-5, y_pred*10**-5))


# #### Calculate vehicle emissions for the grid cells with power plants

# I built a model that predicts motor vehicle emissions for a grid cell. Next, I used the model to predict the motor vehicle emissions in power plant grid cells.

# In[ ]:


#Create a data frame with only the power plant grid cells
ds_powerplants = ds.where(ds.plants_mask == 1)
ds_powerplants = ds_powerplants.drop(['no_plants_mask','plants_mask','land_mask'])
df_powerplants = ds_powerplants.to_dataframe().dropna()
df_powerplants = df_powerplants[['NO2_column_number_density', 'cloud_fraction', 'night_avg_rad', 'gldas_wind_mean', 'gldas_airT_max', 'gldas_pres_mean', 'gldas_lwdown_mean', 'position_lat_id', 'position_lon_id']].copy()
df_powerplants['arcsin_cloud_fraction'] = np.arcsin(np.sqrt(df_powerplants['cloud_fraction']))
df_powerplants['log_night_avg_rad'] = np.log(df_powerplants['night_avg_rad'])
df_powerplants.drop(['cloud_fraction', 'night_avg_rad'], axis=1, inplace=True)


# In[ ]:


X_powerplants = np.array(df_powerplants.drop(['NO2_column_number_density', 'position_lat_id', 'position_lon_id'], axis=1))
y_powerplants = np.array(df_powerplants['NO2_column_number_density'])
y_powerplants = y_powerplants*(10**5) #The numbers are very small so rescaling for model training and prediction
df_powerplants['predict_vehicles'] = model.predict(X_powerplants)*(10**-5)


# The $NO_2$ from motor vehicles is subtracted from the total $NO_2$ in grid cells with power plants.  The excess $NO_2$ is the $NO_2$ produced by power plants. 

# In[ ]:


df_powerplants['NO2_excess'] = df_powerplants['NO2_column_number_density'] - df_powerplants['predict_vehicles']

# In some cases, the predicted vehicle emissions were higher than the measured emissions in a grid cell. 
# The values in these cells were converted to zero in order to quantify the emissions factor
df_powerplants.loc[df_powerplants['NO2_excess']<0, 'NO2_excess'] = 0


# ### Annual emissions factor for Puerto Rico
# The satellite sensor passes over Puerto Rico once per day. Therefore, I make the assumption the residence time for $NO_2$ in a grid cell is approximately 1 day and calculate the annual emissions factor. If the residence time is more or less than 1 day, then a more complicated dynamic model would need to be developed.  

# In[ ]:


#mol/m^2 per day converted to mol/year, a grid cell is approximately 1 km^2
total_emissions = round(df_powerplants['NO2_excess'].sum()*(1000*1000)*365, 2) 
print(total_emissions)


# In[ ]:


#gwh for a year using all sources of power including wind, hydro, and solar
total_generated = round(plants['estimated_generation_gwh'].sum())
print(total_generated)


# In[ ]:


#(mol NO2/year)/(mwh/year) = mol NO2/mwh of fossil fuel generated
emissions_factor = total_emissions/total_generated #(mol NO2/year)/(mwh/year) = mol NO2/mwh of fossil fuel generated
print(round(emissions_factor, 3))


# **The estimated annual emissions factor for Puerto Rico is 0.103 mol $NO_2$ per megawatt hour of power generated.**

# During data exploration, I calculated the maxium potential generation using the capacity_mw.  In some cases, the estimated_generation_gwh for a power plant in the dataset exceeded the maximum potential annual generation, which is not possible. 
# 
# There is an additional reason that I do not think that the estimated_generation_gwh values are correct for individual power plants.  Some of the gas power plants have the exact same estimated_generation_gwh to the 10th decimal place. This is very unlikely. Also, the values could not be that precise.   
# 

# ### Calculate the monthly emissions factor
# One of the objectives is to calculate an emissions factor for each month.  I used the same method that I used to calculate the annual emissions factor.  The wind speed threshold was much higher for the monthly data to get geographic coverage.  Therefore, the emissions facor calculations are likely to be higher. 

# In[ ]:


# Read in data from the saved file created using the code in the regridding section above.
ds_monthly = xr.open_dataset('../input/ds4g-emissions-nc/monthly_ds4g_emissions.nc')


# #### Build model to predict motor vehicle $NO_2$ emissions for each grid cell

# In[ ]:


ds_monthly['landcover_category'] = ds_monthly['landcover_category'].fillna(0)
ds_monthly_vehicles = ds_monthly.where((ds_monthly.land_mask == 1) & 
                   (ds_monthly.no_plants_mask == 1) & 
                   (ds_monthly.landcover_category == 0))
ds_monthly_vehicles = ds_monthly_vehicles.drop(['no_plants_mask','plants_mask','land_mask', 'position_lat_id', 'position_lon_id'])
df_monthly_vehicles = ds_monthly_vehicles.to_dataframe().dropna()


# Extract important features for predicting $NO_2$ emissions.

# In[ ]:


df_monthly_vehicles = df_monthly_vehicles[['NO2_column_number_density', 'cloud_fraction', 'night_avg_rad', 'gldas_wind_mean', 'gldas_airT_max', 'gldas_pres_mean', 'gldas_lwdown_mean']].copy()
df_monthly_vehicles['arcsin_cloud_fraction'] = np.arcsin(np.sqrt(df_monthly_vehicles['cloud_fraction']))
df_monthly_vehicles['log_night_avg_rad'] = np.log(df_monthly_vehicles['night_avg_rad'])
df_monthly_vehicles.drop(['cloud_fraction', 'night_avg_rad'], axis=1, inplace=True)


# In[ ]:


df_monthly_vehicles2 = df_monthly_vehicles.copy()
df_monthly_vehicles2['NO2_column_number_density'] = df_monthly_vehicles['NO2_column_number_density']*(10**5)
sns.pairplot(df_monthly_vehicles2)


# Build a XGBoost model for predicting monthly $NO_2$.

# In[ ]:


X_monthly_vehicles = np.array(df_monthly_vehicles.drop(['NO2_column_number_density'], axis=1))
y_monthly_vehicles = np.array(df_monthly_vehicles['NO2_column_number_density'])
y_monthly_vehicles = y_monthly_vehicles*(10**5) #The numbers are very small so rescaling for model training and prediction
X_monthly_train, X_monthly_test, y_monthly_train, y_monthly_test = train_test_split(X_monthly_vehicles, y_monthly_vehicles, test_size=0.33, random_state=122)
model_monthly = XGBRegressor(learning_rate = 0.01, objective='reg:squarederror', n_estimators = 2000)
model_monthly.fit(X_monthly_train, y_monthly_train)
y_monthly_train_pred = model_monthly.predict(X_monthly_train)
y_monthly_pred = model_monthly.predict(X_monthly_test)


# In[ ]:


plt.scatter(y_monthly_test, y_monthly_pred)  #The numbers are very small, multiply numbers in axes by 10**-5 for actual values


# Evaluate the performance of the model.  The model does not overfit the data. 

# In[ ]:


print(r2_score(y_monthly_train, y_monthly_train_pred))
print(r2_score(y_monthly_test, y_monthly_pred))
print(mean_squared_error(y_monthly_test*10**-5, y_monthly_pred*10**-5))
print(mean_absolute_error(y_monthly_test*10**-5, y_monthly_pred*10**-5))


# #### Calculate vehicle emissions for the grid cells with power plants
# I built a model that predicts motor vehicle emissions for a grid cell. Next, I used the model to predict the motor vehicle emissions in power plant grid cells.

# In[ ]:


ds_monthly_powerplants = ds_monthly.where(ds_monthly.plants_mask == 1) 
ds_monthly_powerplants = ds_monthly_powerplants.drop(['no_plants_mask','plants_mask','land_mask'])
df_monthly_powerplants = ds_monthly_powerplants.to_dataframe().dropna()
df_monthly_powerplants = df_monthly_powerplants[['NO2_column_number_density', 'cloud_fraction', 'night_avg_rad', 'gldas_wind_mean', 'gldas_airT_max', 'gldas_pres_mean', 'gldas_lwdown_mean']].copy()
df_monthly_powerplants['arcsin_cloud_fraction'] = np.arcsin(np.sqrt(df_monthly_powerplants['cloud_fraction']))
df_monthly_powerplants['log_night_avg_rad'] = np.log(df_monthly_powerplants['night_avg_rad'])
df_monthly_powerplants.drop(['cloud_fraction', 'night_avg_rad'], axis=1, inplace=True)


# In[ ]:


X_monthly_powerplants = np.array(df_monthly_powerplants.drop(['NO2_column_number_density'], axis=1))
y_monthly_powerplants = np.array(df_monthly_powerplants['NO2_column_number_density'])
y_monthly_powerplants = y_monthly_powerplants*(10**5) #The numbers are very small so rescaling for model training and prediction
df_monthly_powerplants['predict_vehicles'] = model_monthly.predict(X_monthly_powerplants)*(10**-5)


# The $NO_2$ from motor vehicles is subtracted from the total $NO_2$ in grid cells with power plants.  The excess $NO_2$ is the $NO_2$ produced by power plants. 

# In[ ]:


df_monthly_powerplants['NO2_excess'] = df_monthly_powerplants['NO2_column_number_density'] - df_monthly_powerplants['predict_vehicles']
df_monthly_powerplants[df_monthly_powerplants['NO2_excess']<0] = 0


# The number of days in each month is extracted from the dates in order ot be used to convert the daily emissions to monthly emissions. 

# In[ ]:


monthly_EF = pd.DataFrame(df_monthly_powerplants['NO2_excess'].groupby('time').sum())
monthly_EF['date'] = monthly_EF.index
monthly_EF['days_in_month'] = monthly_EF['date'].dt.day
monthly_EF.drop('date', axis=1, inplace=True)


# Total emissions were calculated for each month. I divided the estimated_generation_gwh by 12 months in order to get the generation per month. However, the estimated_generation likely fluctuates on a monthly basis so monthly power generation data would improve calculations of monthly emissions factors.  

# In[ ]:


#mol/m^2 per day converted to mol/month
monthly_EF['total_emissions'] = round(monthly_EF['NO2_excess']*(1000*1000)*monthly_EF['days_in_month'], 2) 
monthly_EF['total_generated'] = round(plants['estimated_generation_gwh'].sum()/12)
monthly_EF['EF'] = monthly_EF['total_emissions']/monthly_EF['total_generated']


# ### Monthly emissions factor for Puerto Rico

# In[ ]:


monthly_EF


# In[ ]:


plt = monthly_EF.EF.plot()


# **The monthly emissions factors for Puerto Rico ranged from 0.06 to 0.14 from July, 2018 to June, 2019.**

# ### Calculate the marginal emissions factor
# The emissions factor for each type of fossil fuel can be calculated using grid cell locations grouped by primary fuel type.  Some grid cells had both a gas power plant and an oil power plant so these grid cells are labeled as 'Mixed'.

# In[ ]:


plants_fossil = pd.read_csv('../input/ds4g-emissions-nc/plants_fossil.csv')
plants_fossil_sum = plants_fossil[['position_lat', 'position_lon', 'estimated_generation_gwh', 'num_plants']].groupby(['position_lat', 'position_lon'], as_index=False).sum()
plants_fossil_1plant = pd.merge(plants_fossil, plants_fossil_sum[plants_fossil_sum['num_plants']==1])
plants_fossil_1plant = plants_fossil_1plant[['position_lat', 'position_lon', 'primary_fuel']]
df_powerplants_em = df_powerplants[['position_lat_id', 'position_lon_id', 'predict_vehicles', 'NO2_excess']].copy()
df_powerplants_loc = df_powerplants_em.groupby(['position_lat_id', 'position_lon_id'], as_index=False).sum()
df_powerplants_loc[['position_lat_id','position_lon_id']] = df_powerplants_loc[['position_lat_id','position_lon_id']].applymap(np.int64)
df_powerplants_loc.rename({'position_lat_id':'position_lat', 'position_lon_id':'position_lon'}, axis=1, inplace=True)
plants_1type = pd.merge(plants_fossil_sum, df_powerplants_loc, how='left', on=['position_lat','position_lon'])
plants_alltypes = pd.merge(plants_1type, plants_fossil_1plant, how='left', on=['position_lat','position_lon'])
plants_alltypes['primary_fuel'] = plants_alltypes['primary_fuel'].fillna('Mixed')
plants_primaryfuel = plants_alltypes[['primary_fuel','NO2_excess', 'estimated_generation_gwh', 'num_plants']].groupby(['primary_fuel'], as_index=False).sum()
plants_primaryfuel['NO2_excess_annual'] = plants_primaryfuel['NO2_excess']*(1000*1000)*365
plants_primaryfuel['EF'] = plants_primaryfuel['NO2_excess_annual']/plants_primaryfuel['estimated_generation_gwh']


# In[ ]:


plants_primaryfuel


# **It is possible to calculate marginal emissions factors. There is a lot of variability for the different fuels.** 
# 
# In particular, $NO_2$ excess for the coal power plant is very small, and the estimated_generation_gwh is very large.  I calculated the percentage of the estimated_generation_gwh for the coal power plant relative to the total generation for Puerto Rico and found that coal is estimated to generate 92% of the power for Puerto Rico. Therefore, I would expect $NO_2$ emissions for coal to be much higher. I also estimated the fraction of coal compared to the total capacity and found it to be 7% of the total capacity. These calculations do not seem to be in agreement with each other so this may be related to the method that was used to calculate estimated_generation_gwh.  

# ### Conclusions
# Emissions factors were calculated for the sub-national region of Puerto Rico for monthly and yearly time scales.  As more data are collected by the satellite sensor over time, additional analyses can be done to analyze seasonal patterns and trends in $NO_2$ emissions over longer periods of time.  Marginal emissions factors were also calculated but were highly variable and need further investigation.
# 
# The benefit of using a satellite sensor to calculate emissions factors is that data are collected over large geographic areas making it possible to calculate emissions factors over spatial and temporal scales that otherwise would not be possible.  The drawback is that there are many sources of $NO_2$ and attributing the emissions to power plants will have measurement errors that vary with local environmental conditions.   
# 
# **All these calculations of emissions factors need validation.** 
# 

# ### Validation
# These results have **not** been validated with *in situ* measurements for power plants. If data exist, a sub-national region where $NO_2$ emissions are explicitly measured for power plants could be used to validate the results.    
# 

# ### Extension to other sub-national areas
# This method for calculating emissions factors could be applied to other sub-national areas.  All the data sets used were global data sets.  In particular, it would be interesting calculate emissions factors for areas with known power generation and known emissions factors.  The Global Power Plant Database includes power generation for invdividual power plants when it is available.  
# 
# Puerto Rico is an island so there are some unique challenges with measuring emissions in coastal areas where many power plants are located along the land-sea margin.  For example, gldas data are only available for land and did not cover all the areas with power plants in Puerto Rico without extending the data set to surrounding grid cells. Coastal areas also tend to experience daily wind events which advect $NO_2$ emissions away from source grid cells. Therefore, calculating emissions for a region that is not influenced by coastal weather patterns would also be beneficial for evaluating the methodology. 

# ### Additional Thoughts
# Wind advection transports pollutants away from point sources such as a power plants, which makes it difficult to attribute pollutants measured by satellites. I filtered the measurements for the lowest wind speed days in order to calculate the emissions factor for the power plants.  However, this approach excludes a large proportion of the available data.  A pollution transport simulation model which can trace the movement of pollutants back to original sources would be useful for quantifying the total pollutants emitted by a power plant.  This type of model could also be used to evaluate the residence time of $NO_2$ in a grid cell.  This would be particularly useful for calculating higher temporal resolution, months or weeks, emissions factors. I used a much higher wind speed threshold to calculate the monthly emissions factors because the coverage area was too low using a low wind speed threshold. 
# 
# 
# $NO_{2_{Sentinel}} = NO_{2_{powerplants}} + NO_{2_{vehicles}} + NO_{2_{silage}}$ - $NO_{2_{wind}}$
# 
# 
# Ozone is generated when $NO_2$ reacts with VOCs on hot sunny days.  Depending on the rate of conversion $NO_2$ may be underestimated when these conditions occur.  
# 
# $NO_{2_{Sentinel}} = NO_{2_{powerplants}} + NO_{2_{vehicles}} + NO_{2_{silage}}$ - $NO_{2_{wind}}$ - $NO_{2_{ozone}}$
# 
# There are sources of $NO_2$ emissions in addition to power plants, motor vehicles, and silage.  These could also be taken into account. 

# ### Acknowledgements
# 
# This analysis utilized data from the following sources:
# 
# Global Power Plant database by WRI  
# Sentinel 5P OFFL NO2 by EU/ESA/Copernicus  
# Global Forecast System 384-Hour Predicted Atmosphere Data by NOAA/NCEP/EMC  
# Global Land Data Assimilation System by NASA  
# GFSAD1000: Cropland Extent 1km Multi-Study Crop Mask, Global Food-Support Analysis Data by USGS  
# VIIRS Nighttime Day/Night Band Composites Version 1 by NOAA  
# GHRSST Global 1-km Sea Surface Temperature (G1SST) by NOAA  
# GPWv411: Population Density (Gridded Population of the World Version 4.11) by CIESIN  

# In[ ]:




