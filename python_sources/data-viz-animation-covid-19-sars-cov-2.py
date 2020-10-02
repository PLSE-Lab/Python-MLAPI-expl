#!/usr/bin/env python
# coding: utf-8

# Given the dataset, what I envision first is how fast 2019-nCoV (UPDATE: 2020.02.12 newly named as COVID-19 / SARS-CoV-2) spreads in China across different provinces reflected in time. An animation would be pretty straightforward to visualize how severe the situation is.

# <img src="https://imgur.com/Ke50fR6.gif">

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
from datetime import datetime
from IPython.display import Image

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# To output an animation, we need a shapefile where the data can be visualized against. The shapefile is downloaded at [here](https://www.diva-gis.org/gdata).

# # Prepare Data

# In[ ]:


# set the filepath and load in the shapefile using geopandas
fp = '/kaggle/input/china-shape-file/CHN_adm1.shp'
map_df = gpd.read_file(fp, encoding='utf-8')

# check data type so we can see that this is not a normal dataframe, but a GEO dataframe
map_df.head()


# In[ ]:


# rename NAME_1 colum to 'Province/State' for indexing later; extract 'Province/State' and 'geometry' columns as useful information
map_df.rename(index=str, columns={'NAME_1': 'Province/State'}, inplace=True)
map_df = map_df[['Province/State', 'geometry']]

# rename some names of provinces in China to be compatible with the dataset
map_df.replace({'Nei Mongol' : 'Inner Mongolia', 
                'Xinjiang Uygur' : 'Xinjiang', 
                'Ningxia Hui' : 'Ningxia' , 
                'Xizang' : 'Tibet'}, inplace=True)


# In[ ]:


# load the dataset
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')
df.shape


# In[ ]:


# join the geodataframe with the dataset dataframe
merged = map_df.set_index('Province/State').join(df.set_index('Province/State'))
merged.head()


# In[ ]:


# replace 'China' into 'Mainland China' since there is inconsistency in the provided dataset. See the summary above.
merged.replace({'China' : 'Mainland China'}, inplace=True)

# unify the format of datetime
merged.loc[:,'Date'] = pd.to_datetime(merged['Date'], infer_datetime_format=True)
merged.loc[:,'Date'] = merged['Date'].apply(lambda x : datetime.strftime(x, "%m/%d/%Y %H:%M"))
merged.head()


# In[ ]:


# check the date information for the animation later
merged['Date'].sort_values().unique()


# In[ ]:


# pivot the merged dataframe so that the columns represent the confirmed number (used as an example in this case) per each date
merge_pivot = merged.pivot(columns='Date', values='Confirmed')

# reset the index for filling na for next step
merge_pivot.reset_index(inplace=True)

# fill missing values if there are
merge_pivot = merge_pivot.fillna(method='bfill', axis=1)

merge_pivot.head()


# In[ ]:


# set 'Province/State' as index for joining with map_df: geoDataFrame
merge_pivot.set_index('Province/State', inplace=True)
columns = merge_pivot.columns.to_list()

# important since during pivot, data type might be escalated to object, which categorizes the legend of the visualization. 
# Force the type to be int can generate the colorbar for choropleth
merge_pivot.loc[:, columns] = merge_pivot[columns].astype(int)

# join map_df and merge_pivot to get the final geodataframe with the confirmed numbers outlined vertically as col
merged1 = map_df.set_index('Province/State').join(merge_pivot)
merged1.head()


# # Data Visualization & Animation Creation

# In[ ]:


# create a loop to make animations against the date

# save all the maps in the output folder
output_path = '/kaggle/working'

images = []
list_of_last_updates = merged['Date'].sort_values().unique().tolist()

# set the min and max range for the choropleth map
vmin, vmax = 0, 1000

# start the for loop to create one map per year
for date_time in list_of_last_updates:
    
    # create figure and axes for Matplotlib
    fig, ax = plt.subplots(1, figsize=(15, 15))
    
    # remove the axis
    ax.axis('off')
    
    ax.set_title('COVID-19-SARS-CoV-2 confirmed cases in China', fontdict={'fontsize': '20', 'fontweight' : '3'})
    ax.annotate('Date: ' + date_time, xy=(0.6, 0.1), xycoords='figure fraction', fontsize=14, color='#555555')


    fig = merged1.plot(column=date_time, cmap='Reds', linewidth=0.5, edgecolor='white', figsize=(10,10), 
                       legend=True, ax=ax, norm=plt.Normalize(vmin=vmin, vmax=vmax), vmin=vmin, vmax=vmax) 

    filepath = os.path.join(output_path, 'COVID-19-SARS-CoV-2-' + datetime.strptime(date_time, '%m/%d/%Y %H:%M').strftime('%m-%d-%Y-%H-%M') + '.png')

    chart = fig.get_figure()
    chart.savefig(filepath, dpi=300)
    plt.close(chart)
    
    images.append(imageio.imread(filepath))

imageio.mimsave('2019-nCoV.gif', images,duration = 0.5)


# <img src="2019-nCoV.gif">

# In[ ]:


merged1.copy().reset_index().drop(['geometry'], axis=1).to_csv('/kaggle/working/output.csv', index=False)


# In[ ]:




