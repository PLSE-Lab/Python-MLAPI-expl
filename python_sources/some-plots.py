#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import geopandas as gpd
import shapefile as shp
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point, Polygon


# In[ ]:


pd.__version__


# In[ ]:


covid_train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')
covid_test = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')


# In[ ]:


covid_train.head()


# In[ ]:


covid_train.info()


# In[ ]:


sorted_df = covid_train.groupby("Country/Region").agg("sum").sort_values(by=['Fatalities'], ascending=False).reset_index()


# In[ ]:


def div(df_row):
    if df_row.ConfirmedCases == 0:
        return 0
    else:
        return df_row.Fatalities / df_row.ConfirmedCases


# In[ ]:


covid_cnt_by_country = sorted_df[["Country/Region", "ConfirmedCases", "Fatalities"]]
covid_cnt_by_country['DeathRatio'] = covid_cnt_by_country.apply(lambda x: div(x), axis = 1)


# In[ ]:


covid_cnt_by_country.head(20)


# In[ ]:


crs = {'init': 'epsg:4326'}
geometry = [Point(xy) for xy in zip(covid_train['Long'], covid_train['Lat'])]
geo_df = gpd.GeoDataFrame(covid_train, crs = crs, geometry = geometry)


# In[ ]:


fig, ax = plt.subplots(figsize = (25, 20))
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.plot(ax = ax, color = "black")
geo_df.plot(ax = ax, markersize = 5, color = "yellow", marker = "o")


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize = (25, 20))
fig1 = sns.barplot(covid_cnt_by_country["Country/Region"][:10], np.log(covid_cnt_by_country["Fatalities"][:10]), 
                   ax = ax[0], palette=sns.color_palette("GnBu_d"))
fig2 = sns.barplot(covid_cnt_by_country["Country/Region"][:10], covid_cnt_by_country["DeathRatio"][:10], 
                   ax = ax[1], palette=sns.color_palette("GnBu_d"))
fig1.set_xticklabels(fig1.get_xticklabels(), rotation=45)
fig2.set_xticklabels(fig2.get_xticklabels(), rotation=45)
fig1.set(ylabel = "Log Fatalities")
fig1.set_title("Top 10 Log Fatality by Countries", size = 20)
fig2.set_title("Death Ratio by Countries", size = 20)


# In[ ]:





# In[ ]:





# In[ ]:




