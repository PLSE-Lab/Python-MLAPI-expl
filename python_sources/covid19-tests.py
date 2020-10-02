#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import r2_score

from sympy import symbols, diff

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_row', 1000)


# In[ ]:


#!conda install -c conda-forge folium=0.5.0 --yes
import folium

print('Folium installed and imported!')


# In[ ]:


conf = pd.read_csv("/kaggle/input/corona-virus-time-series-dataset/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv")
rip = pd.read_csv("/kaggle/input/corona-virus-time-series-dataset/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv")


# In[ ]:


conf.head()


# In[ ]:


rip.head()


# In[ ]:


conf = conf.drop(columns=['Province/State'])
rip = rip.drop(columns=['Province/State'])
time = conf.drop(columns=['Lat']).drop(columns=['Long'])


# In[ ]:


time.head()


# In[ ]:


conf = conf.rename(columns={"Country/Region": "Country"})
conf.head()


# In[ ]:


rip = rip.rename(columns={"Country/Region": "Country"})
rip.head()


# In[ ]:


time = time.rename(columns={"Country/Region": "Country"})
time.head()


# In[ ]:


time = time.set_index('Country')
time = time.sort_values('Country', ascending=True)
daily = time.transpose()
daily.head()


# In[ ]:


daily=daily.pct_change()
daily.head(20)


# In[ ]:


daily=daily.fillna(0)
daily.head(20)


# In[ ]:


daily = daily.replace([np.inf, -np.inf], 0)
daily.head(20)


# In[ ]:


daily.head()
#daily=daily.index.sort_values()


# In[ ]:


loc = conf[['Country','Lat','Long']]
loc = loc.groupby('Country').max()
loc.head(20)


# In[ ]:


conf.head()


# In[ ]:


rip.head()


# In[ ]:


rip = rip.drop(columns=['Lat','Long'])
conf = conf.drop(columns=['Lat','Long'])


# In[ ]:


rip = rip.groupby(['Country']).sum()
rip.head()


# In[ ]:


conf = conf.groupby(['Country']).sum()
conf.head()


# In[ ]:


conft = conf.transpose()
ript = rip.transpose()
conft = conft.rename(columns={"Country": "Day"})
ript = ript.rename(columns={"Country": "Day"})


# In[ ]:


conft.head()


# In[ ]:


ript.head()


# In[ ]:


plt.figure(figsize=(80,40))
plt.plot(conft.index,conft, alpha=.75)
print('Done!')


# In[ ]:


list(conft)


# In[ ]:


c = input('What country shall we plot?')
rip_plt=ript[c]
daily_plt = daily[c]
conf_plt=conft[c]
daily_plt = daily_plt[daily_plt > 0]
rip_plt = rip_plt[rip_plt != 0]
conf_plt = conf_plt[conf_plt != 0]
d = input('What country shall we compare?')
rip_pt=ript[d]
daily_pt = daily[d]
conf_pt=conft[d]
daily_pt = daily_pt[daily_pt > 0]
rip_pt = rip_pt[rip_pt != 0]
conf_pt = conf_pt[conf_pt != 0]
xx = daily.index


# In[ ]:


plt.figure(figsize=(20,10))
plt.plot(conf_plt.index,conf_plt, color='black',linestyle='--')
plt.plot(rip_plt.index, rip_plt, color='black',linestyle='--')
plt.fill_between(conf_plt.index,conf_plt,color='red', alpha=0.5)
plt.fill_between(rip_plt.index,rip_plt,color='red', alpha=0.5)
plt.plot(conf_pt.index,conf_pt, color='black')
plt.plot(rip_pt.index, rip_pt, color='black')
plt.fill_between(conf_pt.index,conf_pt,color='orange', alpha=0.5)
plt.fill_between(rip_pt.index,rip_pt,color='orange', alpha=0.5)
plt.legend([c, c,d,d])
plt.xticks(rotation=70)
plt.grid(b=None, which='major', axis='both',linestyle='-', linewidth=.5)
#XX = np.arange(0.0, 60.0, 0.1)
#yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2) + clf.coef_[0][3]*np.power(XX, 3) + clf.coef_[0][4]*np.power(XX, 4) + clf.coef_[0][5]*np.power(XX, 5) + clf.coef_[0][6]*np.power(XX, 6)
#plt.plot(rip, conf, '-r' )
#plt.xlabel("Day")
#plt.ylabel("Total")


# In[ ]:


daily_plt=daily_plt.reindex()
daily_pt=daily_pt.reindex()


# In[ ]:


plt.figure(figsize=(20,10))
plt.scatter(daily_plt.index, daily_plt*100, color='red', linestyle='dotted')
plt.scatter(daily_pt.index, daily_pt*100, color='orange', linestyle='dotted')
plt.legend([c,d])
plt.xticks(rotation=70)
plt.grid(b=None, which='major', axis='both',linestyle='-', linewidth=.5)


# In[ ]:


conf['Total'] = conf.sum(axis=1)
rip['Total'] = rip.sum(axis=1)


# In[ ]:


conf.shape


# In[ ]:


rip.shape


# In[ ]:


conf.head()


# In[ ]:


rip.head()


# In[ ]:


loc["Ratio"] = 0
loc.head()


# In[ ]:


for i in range(len(loc)):
    loc.Ratio[i] = 100*(rip.iloc[i]['Total'] / conf.iloc[i]['Total'])
    
loc=loc.sort_values('Ratio', ascending=False)
loc.head()


# In[ ]:


loc.shape


# In[ ]:


print(loc)


# In[ ]:


fig, ax = plt.subplots(figsize=(5,5)) 
sns.heatmap(loc.corr(), 
        xticklabels=loc.corr().columns,
        yticklabels=loc.corr().columns, annot=True, linewidths=0.1, cmap="hsv")
fig.savefig("heatmap.png")


# In[ ]:


conft.head()


# In[ ]:


ript.head()


# In[ ]:


rate = ript.sum(axis=1)/conft.sum(axis=1)*100


# In[ ]:


rate.head()


# In[ ]:


rate = rate.reset_index()
rate = rate.reindex()
#rate = rate.rename(columns={'index':'day'})
#rate = rate.rename(columns={1:'Rate'})
#rate = rate.drop(columns=[0])
#rate = rate.rename(columns={1:'Rate'})


# In[ ]:


for i in range(len(rate)):
    rate.loc[i,'Day']=i


# In[ ]:


rate.head()


# In[ ]:


fig, ax = plt.subplots(figsize=(5,5)) 
sns.heatmap(rate.corr(), 
        xticklabels=rate.corr().columns,
        yticklabels=rate.corr().columns, annot=True, linewidths=0.1, cmap="hsv")
fig.savefig("heatmaprate.png")


# In[ ]:


plt.figure(figsize=(20,10))
plt.plot(rate.Day,rate[0], color='black', alpha=1, linewidth=1, linestyle='--')
plt.fill_between(rate.Day,rate[0],color='red', alpha=.4)
plt.grid(b=None, which='major', axis='both',linestyle='-', linewidth=.5)
#plt.bar(rip_plt.index, rip_plt, color='black')


# In[ ]:


loc = loc.sort_values('Ratio', ascending=False)
loc = loc.reset_index()
loc.head(20)


# In[ ]:


# define the world map
world_map = folium.Map(location=[0,0], zoom_start=2.5)
#folium.Circle(location=[0,0],popup='Hello',radius=1000000,color=None,fill=True,fill_color='crimson').add_to(world_map)
len(loc)


# In[ ]:


for i in range(50):
    folium.Circle(
       location=[loc.iloc[i]['Lat'], loc.iloc[i]['Long']],
       popup=print('Country:',loc.iloc[i]['Country'],'Rate',loc.iloc[i]['Ratio']*100),
       radius=float(loc.iloc[i]['Ratio'])*10000,
       color=None,
       fill=True,
       fill_color='red').add_to(world_map)
print('Done!')


# In[ ]:


world_map


# In[ ]:


# download countries geojson file
get_ipython().system('wget --quiet https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/world_countries.json -O world_countries.json')
    
print('GeoJSON file downloaded!')


# In[ ]:


world_geo = r'world_countries.json' # geojson file

# create a plain world map
cmap = folium.Map(location=[0, 0], zoom_start=2, tiles='Mapbox Bright')


# In[ ]:


cmap.choropleth(
    geo_data=world_geo,
    data=loc,
    columns=['Country', 'Ratio'],
    key_on='feature.properties.name',
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='COVID19 Death Rate'
)

# display map
cmap


# In[ ]:


plt.figure(figsize=(20,10))
plt.bar(loc.Country[0:100],loc.Ratio[0:100], color='red')
plt.xticks(rotation=90)
plt.grid(b=None, which='major', axis='both',linestyle='-', linewidth=.5)


# In[ ]:




