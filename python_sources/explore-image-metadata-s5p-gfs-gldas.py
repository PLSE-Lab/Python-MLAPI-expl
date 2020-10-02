#!/usr/bin/env python
# coding: utf-8

# # Explore Image Metadata (S5P, GFS, GLDAS)

#  - [Sentinel 5P OFFL NO2](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2) by [EU/ESA/Copernicus](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-5p-tropomi/document-library)
#  - [Global Land Data Assimilation System](https://developers.google.com/earth-engine/datasets/catalog/NASA_GLDAS_V021_NOAH_G025_T3H) by NASA
#  - [Global Forecast System 384-Hour Predicted Atmosphere Data ](https://developers.google.com/earth-engine/datasets/catalog/NOAA_GFS0P25) by NOAA/NCEP/EMC

# In[ ]:


import pandas as pd
import rasterio as rio
import os

s5p_file = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20190501T161114_20190507T174400.tif'
gldas_file = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gldas/gldas_20180702_1500.tif'
gfs_file = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/gfs_2019011212.tif'

def preview_meta_data(file_name):
    with rio.open(file_name) as img_filename:
        print('Metadata for: ',file_name)
        print('Bounding Box:',img_filename.bounds)
        print('Resolution:',img_filename.res)
        print('Tags:',img_filename.tags())
        print('More Tags:',img_filename.tags(ns='IMAGE_STRUCTURE'))
        print('Number of Channels =',img_filename.count,'\n')

def return_bands(file_name):
    # adapted from 
    # https://www.kaggle.com/gpoulain/ds4g-eda-bands-understanding-and-gee
    image = rio.open(file_name)
    for i in image.indexes:
        desc = image.descriptions[i-1]
        print(f'{i}: {desc}')


# In[ ]:


preview_meta_data(s5p_file)
preview_meta_data(gldas_file)
preview_meta_data(gfs_file)


# In[ ]:


print('S5P: ','\n')
return_bands(s5p_file)
print('\nGLDAS: ','\n')
return_bands(gldas_file)
print('\nGFS: ','\n')
return_bands(gfs_file)


# S5P_NO2 on Earth Engine: 
# ![S5P_NO2 Bands](https://i.imgur.com/kLxrkQL.png)
# 
# GLDAS on Earth Engine: 
# ![GlDAS Bands](https://i.imgur.com/IzMYWYJ.png)
# 
# GFS on Earth Engine: 
# ![GFS Bands](https://i.imgur.com/WxLiygI.png)
# 
# 

# **S5P_NO2 on Kaggle: **
# - same as Earth Engine (12 bands)
# 
# **GLDAS on Kaggle:**
#  - 12 bands instead of 36 bands
#  - 'LWdown_f_tavg', 'Lwnet_tavg', 'Psurf_f_inst', 'Qair_f_inst', 'Qg_tavg', 'Qh_tavg', 'Qle_tavg', 'Rainf_f_tavg', 'SWdown_f_tavg', 'Swnet_tavg', 'Tair_f_inst', 'Wind_f_inst'
# 
# **GFS on Kaggle:**
# - same as Earth Engine (6 bands)
# 

#  - [Sentinel 5P OFFL NO2](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2) by [EU/ESA/Copernicus](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-5p-tropomi/document-library)
#  - [Global Land Data Assimilation System](https://developers.google.com/earth-engine/datasets/catalog/NASA_GLDAS_V021_NOAH_G025_T3H) by NASA
#  - [Global Forecast System 384-Hour Predicted Atmosphere Data ](https://developers.google.com/earth-engine/datasets/catalog/NOAA_GFS0P25) by NOAA/NCEP/EMC

# **More getting started material is available here:**
# * [How to get started with the Earth Engine data](https://www.kaggle.com/paultimothymooney/how-to-get-started-with-the-earth-engine-data/) 
#  - connect to earthengine-API, load and preview data, etc
# * [Overview of the EIE Analytics Challenge](https://www.kaggle.com/paultimothymooney/overview-of-the-eie-analytics-challenge) 
#  - submission instructions, evaluation criteria, etc

# In[ ]:




