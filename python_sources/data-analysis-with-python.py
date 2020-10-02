#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from datetime import timedelta

# Data Visualization Liraries
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go
from IPython.display import display, Markdown

#hide warnings
import warnings
warnings.filterwarnings('ignore')
pyo.init_notebook_mode()

#display max columns of pandas dataframe
pd.set_option('display.max_columns', None)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


cov = pd.read_csv('../input/uncover/UNCOVER_v4/UNCOVER/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-over-time.csv')
cov_country = pd.read_csv('../input/uncover/UNCOVER_v4/UNCOVER/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-by-country.csv')


# In[ ]:


cov_country.head()


# In[ ]:


cov.head()


# # Field description
# * Country_Region - The name of the Country
# * Last_Update - The most recent date the file was pushed
# * Lat - Latitude
# * Long_ - Longitude
# * Confirmed - Aggregated confirmed case count for the state
# * Deaths - Aggregated Death case count for the state
# * Recovered - Aggregated Recovered case count for the state
# * Active - Aggregated confirmed cases that have not been resolved (Active = Confirmed - Recovered - Deaths)
# * Incident_Rate - confirmed cases per 100,000 persons
# * Mortality_Rate - Number recorded deaths * 100/ Number confirmed cases

# In[ ]:


# Helper Function - Missing data check
def missing_data(data):
    missing = data.isnull().sum()
    available = data.count()
    total = (missing + available)
    percent = (data.isnull().sum()/data.isnull().count()*100).round(4)
    return pd.concat([missing, available, total, percent], axis=1, keys=['Missing', 'Available', 'Total', 'Percent']).sort_values(['Missing'], ascending=False)


# In[ ]:


missing_data(cov_country)


# > Dropping off columns with 100% missing values and unnecessary ones

# In[ ]:


cov_country = cov_country.drop(['people_tested','people_hospitalized','iso3'],axis = 1)


# > Filtering out the rows with missing latitude and longitude data.

# In[ ]:


cov_country[cov_country.lat.isnull()]


# * It seems both these data points belong to cruise ships and not necessarily associated with any country
# * Hence dropping these rows temporarily

# In[ ]:


covid_country = cov_country.dropna()


# # Till date global cases reported

# In[ ]:


new_df = pd.DataFrame(covid_country[["confirmed","deaths","recovered","active"]].sum()).transpose()
new_df['mortality_rate'] = covid_country['mortality_rate'].mean()
new_df['incident_rate'] = covid_country['incident_rate'].mean()
new_df


# In[ ]:


import folium
from folium.plugins import MarkerCluster
#empty map
world_map= folium.Map(tiles="cartodbpositron")
marker_cluster = MarkerCluster().add_to(world_map)
#for each coordinate, create circlemarker of user percent
for i in range(len(covid_country)):
        lat = covid_country.iloc[i]['lat']
        long = covid_country.iloc[i]['long']
        radius=5
        popup_text = """Country : {}<br>
                    Confimed : {}<br>
                    Deaths : {}<br>
                    Recovered : {}<br>"""
        popup_text = popup_text.format(covid_country.iloc[i]['country_region'],
                                   covid_country.iloc[i]['confirmed'],
                                       covid_country.iloc[i]['deaths'],
                                       covid_country.iloc[i]['recovered']
                                   )
        folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True).add_to(marker_cluster)
#show the map
world_map


# In[ ]:


fig = px.choropleth(covid_country, locations="country_region",
                    color=covid_country["confirmed"], 
                    hover_name="country_region", 
                    hover_data=["deaths"],
                    locationmode="country names")

fig.update_layout(title_text="Confirmed Cases Heat Map (Log Scale)")
fig.update_coloraxes(colorscale="blues")

fig.show()


# In[ ]:


# Top 20 countries with highest confirmed cases
covid_country_top20=covid_country.sort_values("confirmed",ascending=False).head(20)

fig = px.bar(covid_country_top20, 
             x="country_region",
             y="confirmed",
             orientation='v',
             height=800,
             title='Top 20 countries with COVID19 Confirmed Cases',
            color='country_region')
fig.show()


# In[ ]:


# Top 20 countries with highest active cases
covid_country_top20=covid_country.sort_values("active",ascending=False).head(20)
fig = px.bar(covid_country_top20, 
             x="country_region",
             y="active",
             orientation='v',
             height=800,
             title='Top 20 countries with COVID19 Active Cases',
            color='country_region')
fig.show()


# In[ ]:


# Top 20 countries with highest recovered cases
covid_country_top20=covid_country.sort_values("recovered",ascending=False).head(20)
fig = px.bar(covid_country_top20, 
             x="country_region",
             y="recovered",
             orientation='v',
             height=800,
             title='Top 20 countries with COVID19 Recovered Cases',
            color='country_region')
fig.show()


# # correlation between dimensions

# In[ ]:


corr= covid_country.corr()
plt.figure(figsize=(16,16))
sns.heatmap(corr,cmap="YlGnBu",annot=True)

