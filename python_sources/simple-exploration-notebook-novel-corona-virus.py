#!/usr/bin/env python
# coding: utf-8

# ![](https://www.sciencealert.com/images/2020-01/processed/coronavirus_topic_header_1024.jpg)
# 
# 
# 
# **What is coronavirus?**
# Coronaviruses are a family of viruses known for containing strains that cause potentially deadly diseases in mammals and birds. In humans they're typically spread via airborne droplets of fluid produced by infected individuals.
# 
# Some rare but notable strains, including Wuhan coronavirus (2019-nCoV), and those responsible for severe acute respiratory syndrome (SARS) and Middle East respiratory syndrome (MERS), can cause death in humans.
# 
# Wuhan coronavirus was first identified in the Chinese city of Wuhan in 2019. At the time of writing, numbers of infected are still on the rise, with a number of deaths having been reported. 
# 
# *so here in the data i have a plan of making analysis in china alone and other than china citys ..!! *

# In[ ]:


# import the necessary libraries
import numpy as np 
import pandas as pd 

# Visualisation libraries
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
color = sns.color_palette()
sns.set()
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objs as go
import plotly.offline as py
import pycountry
py.init_notebook_mode(connected=True)
import folium 
from folium import plugins

# Graphics in retina format 
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

# Increase the default plot size and set the color scheme
plt.rcParams['figure.figsize'] = 8, 5
#plt.rcParams['image.cmap'] = 'viridis'


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data = pd.read_csv('../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')
data.head()


# In[ ]:


data.info()


# In[ ]:


data['Last Update'] = data['Last Update'].apply(pd.to_datetime)
data.drop(['Sno'],axis = 1,inplace =True)

data.head()


# In[ ]:


data['Day'] = data['Last Update'].apply(lambda x:x.day)
data['Hour'] = data['Last Update'].apply(lambda x:x.hour)


# In[ ]:


plt.figure(figsize=(16,6))
sns.barplot(x='Day',y='Confirmed',data=data)
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x="Deaths", data=data, color=color[1])
plt.ylabel('Count', fontsize=12)
plt.xlabel('Deaths freq', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Deaths by Corona", fontsize=15)
plt.show()


# In[ ]:


# Creating a dataframe with total no of cases for every country

cases = pd.DataFrame(data.groupby('Country')['Confirmed'].sum())
cases['Country'] = cases.index
cases.index=np.arange(1,32)

global_cases = cases[['Country','Confirmed']]
#global_cases.sort_values(by=['Confirmed'],ascending=False)

plt.figure(figsize=(12,8))
sns.barplot(global_cases.Country, global_cases.Confirmed, alpha=0.8, color=color[2])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Maximum order number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# * **this confirm that Mainland China is the most affected country**

# In[ ]:


case = pd.DataFrame(data.groupby('Province/State')['Confirmed'].sum())
case['Province/State'] = case.index

globel_case = case[['Province/State','Confirmed']]


plt.figure(figsize=(12,8))
sns.barplot(globel_case['Province/State'], globel_case.Confirmed, alpha=0.8, color=color[2])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Maximum order number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# * **Hubei is the state which is more affected by the Coonavirus which is in Mainland China**

# In[ ]:


fig=plt.subplots(figsize=(10,10))
plt.title('Dependence between Confirmed and Deaths',fontsize=12)
sns.regplot(x='Confirmed', y='Deaths',
            ci=None, data=data)
sns.kdeplot(data.Confirmed,data.Deaths)
plt.show()


# * **this states that there is a good relation between attributes Confirmed and the Deaths so we can say a state or a country with more Confirmed cases will be with more Deaths cases**

# In[ ]:


data['date'] = pd.to_datetime(data['Last Update']).dt.date
date_con = data.groupby('date')['Confirmed','Deaths','Recovered'].sum().reset_index()


from plotly.subplots import make_subplots
fig = make_subplots(rows=1, cols=3, subplot_titles=("Comfirmed", "Deaths", "Recovered"))

trace1 = go.Scatter(
                x=date_con['date'],
                y=date_con['Confirmed'],
                name="Confirmed",
                line_color='orange',
                opacity=0.8)
trace2 = go.Scatter(
                x=date_con['date'],
                y=date_con['Deaths'],
                name="Deaths",
                line_color='red',
                opacity=0.8)

trace3 = go.Scatter(
                x=date_con['date'],
                y=date_con['Recovered'],
                name="Recovered",
                line_color='green',
                opacity=0.8)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)
fig.update_layout(template="ggplot2",title_text = '<b>Global Spread of the Coronavirus Over Time </b>',
                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))
fig.show()


# * **here we can see the Confirmed,Deaths and Recovered cases with the dates. This is a increase of the Confirmed cases within 2 days**

# In[ ]:


gro_df = data.groupby(['Country','Province/State']).sum()
gro_df.head(20)


# In[ ]:


grouped_df = data.groupby(["Country", "Province/State"])["Deaths"].aggregate("count").reset_index()

fig, ax = plt.subplots(figsize=(12,20))
ax.scatter(grouped_df['Deaths'].values, grouped_df['Province/State'].values)
for i, txt in enumerate(grouped_df["Country"].values):
    ax.annotate(txt, (grouped_df['Deaths'].values[i], grouped_df['Province/State'].values[i]), rotation=45, ha='center', va='center', color='green')
plt.xlabel('Reorder Ratio')
plt.ylabel('department_id')
plt.title("Reorder ratio of different aisles", fontsize=15)
plt.show()


# * **So the Data says that there are more affected cases in Mainland Chain so we will do analysis in this way firstly we separate the data of Mainland chain and make analysis an then we separate the data other than the china country and make analysis.. so that we can understand the data in a very cleary way**

# In[ ]:


# Make a data frame with dots to show on the map
world_d = dict(
   name=list(global_cases['Country']),
    lat=[-25.27,12.57,56.13,61.92,46.23,51.17,22.32,20.59,41.87,36.2,22.2,35.86,4.21,28.39,12.87,1.35,35.91,7.87,23.7,15.87,37.09,23.42,14.06,],
   lon=[133.78,104.99,-106.35,25.75,2.21,10.45,114.17,78.96,12.56,138.25,113.54,104.19,101.98,84.12,121.77,103.82,127.77,80.77,120.96,100.99,-95.71,53.84,108.28],
   Confirmed=list(global_cases['Confirmed'])
)
world_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in world_d.items() ]))
world_data = world_data.fillna(method='ffill') 


# create map and display it
world_map = folium.Map(location=[10, -20], zoom_start=2.3,tiles='Stamen Toner')

for lat, lon, value, name in zip(world_data['lat'], world_data['lon'], world_data['Confirmed'], world_data['name']):
    folium.CircleMarker([lat, lon],
                        radius=10,
                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Confirmed Cases on 30th Jan 2020</strong>: ' + str(value) + '<br>'),
                        color='red',
                        
                        fill_color='red',
                        fill_opacity=0.7 ).add_to(world_map)
world_map


# * **here in the above map all the red marks states that the affected country**

# In[ ]:


df_china = data[data['Country'] == "Mainland China"]
df_china.head()


# * **data separated only the Mainland China**

# In[ ]:


plt.figure(figsize=(10,10))
sns.barplot(x='Day',y='Confirmed',data=df_china)
plt.show()


# * **this says that within 2 days they are more confirmed cases **

# In[ ]:


plt.figure(figsize=(10,10))
cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
sns.scatterplot(x="Confirmed", y="Deaths",
                     hue="Day", size="Hour",
                     palette=cmap, sizes=(10, 200),
                     data=df_china)
plt.show()


# * **so this says that they are more Confirmed and Deaths cases within 10days and less than 16hr**

# In[ ]:


deth_con = df_china.groupby('Province/State')['Deaths'].sum().reset_index().sort_values(by=['Deaths'],ascending=False)
plt.figure(figsize=(12,8))
sns.pointplot(deth_con['Province/State'].values, deth_con['Deaths'].values, alpha=0.8, color=color[2])
plt.xticks(rotation='vertical')
plt.show()


# * **when we see the states in the Mainland Chain Hubei is the most affected state the above visual proves that**

# In[ ]:


grouped_df = df_china.groupby(['Province/State'])['Recovered'].aggregate('count').reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_df['Province/State'].values, grouped_df['Recovered'].values, alpha=0.8, color=color[2])
plt.ylabel('Recovered count', fontsize=12)
plt.xlabel('state of mainland china', fontsize=12)
plt.title("Confirmed in state of mainland china", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


# * **seem that other than Tibet all other states have good Recovery**
# 
# * **lets do some more deep analysis lets separate the data with having only state Hubei information and make some visualization to understand the data **

# In[ ]:


df_hubei = data[data['Province/State'] == "Hubei"]
df_hubei


# In[ ]:


plt.figure(figsize=(10,10))
sns.barplot(x='Day',y='Confirmed',data=df_hubei)
plt.show()


# In[ ]:


df_hubei.groupby(['Province/State'])['Deaths'].aggregate('sum').reset_index().sort_values(by=['Deaths'],ascending=False)


# * **they are totally 1653 Death cases in just one state**

# In[ ]:


f, ax = plt.subplots(figsize=(15, 10))


sns.barplot(x="Confirmed", y="Province/State", data=df_china[1:],
            label="Confirmed", color="orange",alpha=0.7)


sns.barplot(x="Recovered", y="Province/State", data=df_china[1:],
            label="Recovered", color="g",alpha=0.7)


sns.barplot(x="Deaths", y="Province/State", data=df_china[1:],
            label="Deaths", color="r",alpha=0.7)

# Add a legend and informative axis label
ax.set_title('Confirmed vs Recovered vs Death in mainland China', fontsize=20, fontweight='bold', position=(0.53, 1.05))
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 24), ylabel="",
       xlabel="Stats")
sns.despine(left=True, bottom=True)


# * **here the above visualization states that the affeated state in china is Hudei and all most all the states in the dataset have confirmed cases more and Zhejiang is the most recovered state in Mainland Chian**
# 
# 
# 
# * **Next task is to do understanding the other than chian so lets separate the data and make some visualization to understand the data **

# In[ ]:


df_other_than_china = data[(data['Country'] != 'China') & (data['Country'] != 'Mainland China')]
df_other_than_china


# In[ ]:


df_other_than_china.groupby(['Province/State'])['Confirmed'].aggregate('sum').reset_index().sort_values(by=['Confirmed'],ascending=False).head()


# * **So this says that the other than china state Hong Kong is with most Confirmed cases**

# In[ ]:


grouped_df = df_other_than_china.groupby(['Province/State'])['Confirmed'].aggregate('count').reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_df['Province/State'].values, grouped_df['Confirmed'].values, alpha=0.8, color=color[2])
plt.ylabel('Confirmed count', fontsize=12)
plt.xlabel('Province/State other than china', fontsize=12)
plt.title("Confirmed in Province/State of other than china", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(15, 10))


sns.barplot(x="Confirmed", y="Country", data=df_other_than_china[1:],
            label="Confirmed", color="orange",alpha=0.7)


sns.barplot(x="Recovered", y="Country", data=df_other_than_china[1:],
            label="Recovered", color="g",alpha=0.7)


sns.barplot(x="Deaths", y="Country", data=df_other_than_china[1:],
            label="Deaths", color="r",alpha=0.7)

# Add a legend and informative axis label
ax.set_title('Confirmed vs Recovered vs Death other than China', fontsize=20, fontweight='bold', position=(0.53, 1.05))
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 24), ylabel="",
       xlabel="Stats")
sns.despine(left=True, bottom=True)


# * **So this states that the thailand stands with the most confirmed cases and the philippines is the only country with some Death cases and same thailand is with more recovered country**

# In[ ]:


plt.figure(figsize=(12,8))
sns.violinplot(x='Day', y='Confirmed', data=df_other_than_china)
plt.xlabel('Day', fontsize=12)
plt.ylabel('Confirmed', fontsize=12)
plt.show()


# * **Seem like for all the States and country frist 2days are with more Confirmed cases**

# In[ ]:


date_c = df_other_than_china.groupby('date')['Confirmed','Deaths','Recovered'].sum().reset_index()


from plotly.subplots import make_subplots
fig = make_subplots(rows=1, cols=3, subplot_titles=("Comfirmed", "Deaths", "Recovered"))

trace1 = go.Scatter(
                x=date_c['date'],
                y=date_c['Confirmed'],
                name="Confirmed",
                line_color='orange',
                opacity=0.8)
trace2 = go.Scatter(
                x=date_c['date'],
                y=date_c['Deaths'],
                name="Deaths",
                line_color='red',
                opacity=0.8)

trace3 = go.Scatter(
                x=date_c['date'],
                y=date_c['Recovered'],
                name="Recovered",
                line_color='green',
                opacity=0.8)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)
fig.update_layout(template="ggplot2",title_text = '<b>Global Spread of the Coronavirus Over Time other than china </b>',
                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))
fig.show()


# * **And finally here we can see the confirmed,Deaths and Recovered cases with dates in other than the china country **

# **More to come,will soon update with more good understanding and visualization**
# 
# **your one upvote can make me to work more :)**
# 
# * **So please upvote thankyou**
