#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib as mpl
import matplotlib.pyplot as plt
print("Done!")


# In[ ]:


file =  'https://cocl.us/datascience_survey_data'
df = pd.read_csv(file)
df


# # First Task
# A survey was conducted to gauge an audience interest in different data science topics, namely:
# 
# Big Data (Spark / Hadoop)
# Data Analysis / Statistics
# Data Journalism
# Data Visualization
# Deep Learning
# Machine Learning
# The participants had three options for each topic: Very Interested, Somewhat interested, and Not interested. 2,233 respondents completed the survey.
# 
# The survey results have been saved in a csv file and can be accessed through this link: https://cocl.us/datascience_survey_data.
# 
# If you examine the csv file, you will find that the first column represents the data science topics and the first row represents the choices for each topic.
# 
# Use the pandas read_csv method to read the csv file into a pandas dataframe, that looks like the following:
# 
# 
# In order to read the data into a dataframe like the above, one way to do that is to use the index_col parameter in order to load the first column as the index of the dataframe. Here is the documentation on the pandas read_csv method: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
# 
# Once you have succeeded in creating the above dataframe, please upload a screenshot of your dataframe with the actual numbers. (5 marks)

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
df.sort_values(by=['Very interested'], inplace=True, ascending=False)
df.rename(columns={'Unnamed: 0':'Topic'},inplace=True)
df_perc = df[['Topic']]
df_perc = df_perc.join((df[['Very interested','Somewhat interested','Not interested']]/2233)*100)
df_perc.set_index('Topic', inplace=True)
df_perc.round(2)


# # Second Task
# 
# Use the artist layer of Matplotlib to replicate the bar chart below to visualize the percentage of the respondents' interest in the different data science topics surveyed.
# 
# To create this bar chart, you can follow the following steps:
# 
# - Sort the dataframe in descending order of Very interested.
# - Convert the numbers into percentages of the total number of respondents. Recall that 2,233 respondents completed the survey. Round percentages to 2 decimal places.
# As for the chart:
# - use a figure size of (20, 8),
# - bar width of 0.8,
# - use color #5cb85c for the Very interested bars, color #5bc0de for the Somewhat interested bars, and color #d9534f for the Not interested bars,
# - use font size 14 for the bar labels, percentages, and legend,
# - use font size 16 for the title, and,
# - display the percentages above the bars as shown above, and remove the left, top, and right borders.
# Once you are satisfied with your chart, please upload a screenshot of your plot. (5 marks)

# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels =['Data Analysis / Statistics','Machine Learning','Data Visualization','Big Data (Spark / Hadoop)','Deep Learning','Data Journalism']
very_int = df_perc['Very interested']
some_int = df_perc['Somewhat interested']
not_int = df_perc['Not interested']

ind = np.arange(len(very_int))  
width = 0.3

fig, ax = plt.subplots(figsize=(20,8))
rects1 = ax.bar(ind - width, very_int, width, label='Very interested', color='#5cb85c')
rects2 = ax.bar(ind, some_int, width, label='Somewhat interested', color='#5bc0de')
rects3 = ax.bar(ind + width, not_int, width, label='Notr interested', color='#d9534f')

ax.set_title("Percentage of Respondents' Interest In Data Science Areas", fontsize=16)
ax.set_xticks(ind)
ax.set_xticklabels((labels))
ax.get_yaxis().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=14)


def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height().round(2)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom', fontsize=14)


autolabel(rects1, "center")
autolabel(rects2, "center")
autolabel(rects3, "center")

fig.tight_layout()

plt.show()


# # Third Task
# 
# In the final lab, we created a map with markers to explore crime rate in San Francisco, California. In this question, you are required to create a Choropleth map to visualize crime in San Francisco.
# 
# Before you are ready to start building the map, let's restructure the data so that it is in the right format for the Choropleth map. Essentially, you will need to create a dataframe that lists each neighborhood in San Francisco along with the corresponding total number of crimes.
# 
# Based on the San Francisco crime dataset, you will find that San Francisco consists of 10 main neighborhoods, namely:
# 
# - Central,
# - Southern,
# - Bayview,
# - Mission,
# - Park,
# - Richmond,
# - Ingleside,
# - Taraval,
# - Northern, and,
# - Tenderloin.
# 
# Convert the San Francisco dataset, which you can also find here, https://cocl.us/sanfran_crime_dataset, into a pandas dataframe, like the one shown below, that represents the total number of crimes in each neighborhood.
# 
# 
# Once you are happy with your dataframe, upload a screenshot of your pandas dataframe. (5 marks)

# In[ ]:


file =  'https://cocl.us/sanfran_crime_dataset'
df_sf = pd.read_csv(file)
df_sf.head()


# In[ ]:


df_sf_neigh = df_sf.groupby(["PdDistrict"]).count().reset_index()
df_sf_neigh.drop(df_sf_neigh.columns.difference(['PdDistrict','IncidntNum']), 1, inplace=True)
df_sf_neigh.rename(columns={'PdDistrict':'Neighborhood','IncidntNum':'Count'}, inplace=True)
df_sf_neigh


# # Fourth Task
# 
# Now you should be ready to proceed with creating the Choropleth map.
# 
# As you learned in the Choropleth maps lab, you will need a GeoJSON file that marks the boundaries of the different neighborhoods in San Francisco. In order to save you the hassle of looking for the right file, I already downloaded it for you and I am making it available via this link: https://cocl.us/sanfran_geojson.
# 
# For the map, make sure that:
# 
# - it is centred around San Francisco,
# - you use a zoom level of 12,
# - you use fill_color = 'YlOrRd',
# - you define fill_opacity = 0.7,
# - you define line_opacity=0.2, and,
# - you define a legend and use the default threshold scale.
# 
# Once you are ready to submit your map, please upload a screenshot of your Choropleth map. (5 marks)

# In[ ]:


get_ipython().system('wget --quiet https://cocl.us/sanfran_geojson')
get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')
import folium

print('Folium installed and imported!')
print('GeoJSON file downloaded!')


# In[ ]:


sf_geo = 'https://cocl.us/sanfran_geojson'

sf_latitude = 37.77
sf_longitude = -122.42
sf_map = folium.Map(location=[sf_latitude,sf_longitude], zoom_start=12)


# In[ ]:


sf_map.choropleth(
    geo_data=sf_geo,
    data=df_sf_neigh,
    columns=['Neighborhood', 'Count'],
    key_on='feature.properties.DISTRICT',
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Crime Rate per District in San Francisco')

sf_map


# In[ ]:




