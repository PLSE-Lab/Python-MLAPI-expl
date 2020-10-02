#!/usr/bin/env python
# coding: utf-8

# # COVID19-India-Analysis [Kaggle Notebook](https://www.kaggle.com/samacker77k/covid19-india-analysis)
# A notebook dedicated to data visualization and analysis of COVID19 Pandemic in India.
# 
# ---
# 
# This notebook visualizes the effects of COVID19 pandemic in India to help understand the effect of the outbreak demographically.
# 
# Maintained by:
# * Shivani Tyagi [LinkedIn](https://www.linkedin.com/in/shivani-tyagi-09/) [Github](https://github.com/shivitg)
# * Nitika Kamboj [LinkedIn](https://linkedin.com/in/nitika-kamboj) [Github](https://github.com/nitika-kamboj)
# * Samar Srivastava [LinkedIn](https://linkedin.com/in/samacker77l) [Github](https://github.com/samacker77)
#  
# 

# <p style="color:red">Since the API that was previously being used to fetch the data has now been revoked. We will be updating the dataset every 24 hours.</p>

# ---
# ### Coronavirus 2019(COVID-19) 
# #### COVID 19 is an infectious spreading disease,which is casued by severe acute respiratory syndrome coronavirus 2(SARS-Cov-2).This disease was first found in 2019 in Wuhan distirct of China, and is spreading tremendously across the globe,resulted in pandemic declaration by World Health Organization.
# ---

# In[ ]:


import datetime


# <h4 style="color:green">Last update on</h4>

# In[ ]:


now = datetime.datetime.now()

print(now)


# In[ ]:


get_ipython().system('python3 -m pip install folium')


# In[ ]:


get_ipython().system('pip install ner-d')


# ### Importing libraries
# ---

# In[ ]:


import requests
import pandas as pd
import logging
import datetime
from  geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import math
import folium
import numpy as np
import nltk
from nerd import ner
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree 
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML
from scipy import stats
import warnings
import csv

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(font_scale=1.4)


# In[ ]:


def load_data():
    df = pd.read_csv('../input/covid19india/data.csv')
    return df


# In[ ]:


data = load_data()


# ---
# #### Now we have fetched data successfully. Now we will inspect the data. 

# In[ ]:


print("Data Shape ~ Rows = {} | Columns = {}".format(data.shape[0],data.shape[1]))


# In[ ]:


data.columns


# In[ ]:


data.head()


# ### COVID19 Confirmed cases current location
# > Click on the map and counts to play with the graph.

# In[ ]:


def get_lat_long(string):       
    return string.replace('SRID=4326;POINT ','').strip(')(').split()

data["current_location_pt"]=data['current_location_pt'].fillna(data['current_location_pt'].mode().iloc[0])
data['current_loc_latlong'] = data['current_location_pt'].apply(get_lat_long)

data['current_lat'] = data['current_loc_latlong'].apply(lambda x:x[0])
data['current_long'] = data['current_loc_latlong'].apply(lambda x:x[1])

data.current_lat = data.current_lat.astype('float64')
data.current_long = data.current_long.astype('float64')

m_3 = folium.Map(location=[12.4996, 74.9869], tiles='cartodbpositron', zoom_start=4)

# Add points to the map
mc = MarkerCluster()
for idx, row in data.iterrows():
    if not math.isnan(row['current_long']) and not math.isnan(row['current_lat']):
        mc.add_child(Marker([row['current_long'], row['current_lat']]))
m_3.add_child(mc)

# Display the map
m_3


# > On first look we see that the attributes 'id' and 'unique_id' are same. So we check if they have any values that are different.

# #### Checking dtypes

# In[ ]:


data.dtypes


# In[ ]:


data[data['id'] == data['unique_id']].shape


# > Since we have same values in both columns. We can drop one of them and make another as the index

# In[ ]:


data.drop('unique_id',axis=1,inplace=True)


# In[ ]:


data.set_index('id',inplace=True)


# In[ ]:


print("Data Shape ~ Rows = {} | Columns = {}".format(data.shape[0],data.shape[1]))


# In[ ]:


data.head()


# > Since government_id is of no use. We can drop it.

# In[ ]:


data['government_id'].isna().sum()


# In[ ]:


data.drop('government_id',axis=1,inplace=True)


# #### Now we convert date columns to datetime objects since they are in string.

# In[ ]:


data.dtypes


# In[ ]:


date_columns = ['diagnosed_date','status_change_date']


# In[ ]:


for column in date_columns:
    data[column] = pd.to_datetime(data[column])


# In[ ]:


data.head()


# #### Now the data is ready for analysis and preprocessing

# In[ ]:


#Checking null values
data.isna().sum()


# > Imputing missing values with 'Unknown'

# In[ ]:


for i in data.columns:
    if data[i].dtype == 'object':
        data[i] = data[i].fillna('Unknown')


# In[ ]:


data.dtypes


# In[ ]:


data.isna().sum()


# ---
# ### Now the data is ready for EDA
# > Understanding the involved factors in the growth of the Corona Virus via visualization
# ---

# In[ ]:


print("Number of Cities Affected from COVID19: ", data['detected_city'].nunique())
print("#----------------------------------------------------------------------------------------#")
print("Cities Affected: ", data[data['detected_city'].isna()==False]['detected_city'].unique())


# ---
# #### Analysis: 
# #### There are total in total 4000 cities in India. Number of cities affected currently are "155".
# ---

# In[ ]:


print("Number of States Affected from COVID19: ", data['detected_state'].nunique())
print("#----------------------------------------------------------------------------------------#")
print("States Affected: ", data[data['detected_state'].isna()==False]['detected_state'].unique())


# ---
# #### Analysis:
# #### India is a federal union comprising 28 states and 8 union territories, for a total of 36 entities. The current affected entities are "26".
# ---

# #### Let's observe the age factor for coronavirus spread
# ---

# In[ ]:


plt.figure(figsize=(20,15))
data[['age']].plot(kind='hist',bins=[0,20,40,60,80,100],rwidth=1.8)
plt.ylabel('Count of cases reported')
plt.xlabel('Age Group')
plt.show()


# ---
# #### Analysis:
# #### The most common affected people belongs to age group of 20-60.
# ---

# > Graph between the count of affected people and Nationality.

# In[ ]:


data['nationality'].value_counts()


# In[ ]:


plt.figure(figsize=(10,5))
df=pd.DataFrame({'nationality':data['nationality'].value_counts().index,'Count':data['nationality'].value_counts().values})
ax=sns.barplot(x="Count",y="nationality",data=df, palette=sns.dark_palette("blue", reverse=True))
for i in ax.patches:
    ax.text(i.get_width()+0.50, i.get_y()+0.50,             str(int(i.get_width())), fontsize=15,color='black')
ax.set_title('Count of people affected and their nationality', pad=20)
plt.xlabel('Count of people affected', fontsize=20)
plt.ylabel('Nationality', fontsize=20)
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=8)
plt.show()


# > Graph between gender and count of affected people.

# In[ ]:


plt.figure(figsize=(6,5))
ax=sns.barplot(data['gender'].value_counts().index,data['gender'].value_counts().values, palette=sns.dark_palette("blue", reverse=True))
for i in ax.patches:
    ax.text(i.get_x()+.20, i.get_height()+.9,             str(int(i.get_height())), fontsize=15,
                color='black')
ax.set_title('Count of people affected and their Gender', pad=20)
plt.ylabel('Count of people affected', fontsize=20)
plt.xlabel('Gender', fontsize=20)
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=8)
plt.show()


# > Graph between different states and number of detected cases.

# In[ ]:


plt.figure(figsize=(10,10))
ax=sns.barplot(data['detected_state'].value_counts().values,data['detected_state'].value_counts().index, palette=sns.dark_palette("blue", reverse=True))
for i in ax.patches:
    ax.text(i.get_width()+0.50, i.get_y()+0.50,             str(int(i.get_width())), fontsize=12,color='black')
ax.set_title('Count of detected cases in different states of India.', pad=20)
plt.xlabel('Count of people detected', fontsize=20)
plt.ylabel('State', fontsize=20)
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=15)
plt.show()


# #### The graph clearly shows that Maharashtra is the most affected state followed by Kerala 

# > Graph displaying status and count of affected people.

# In[ ]:


plt.figure(figsize=(8,5))
ax=sns.barplot(data['current_status'].value_counts().index,data['current_status'].value_counts().values, palette=sns.dark_palette("blue", reverse=True))
for i in ax.patches:
    ax.text(i.get_x()+.20, i.get_height()+.10,             str(int(i.get_height())), fontsize=15,
                color='black')
ax.set_title('Status of affected people.', pad=20)
plt.ylabel('Count of people', fontsize=20)
plt.xlabel('Status', fontsize=20)
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=8)
plt.show()


# > No. of people affected district wise

# In[ ]:


plt.figure(figsize=(25,20))
ax=sns.barplot(data['detected_district'].value_counts()[:10].values,data['detected_district'].value_counts()[:10].index, palette=sns.dark_palette("blue", reverse=True))
for i in ax.patches:
    ax.text(i.get_width()+0.50, i.get_y()+0.50,             str(int(i.get_width())), fontsize=30,color='black')
ax.set_title('Top 10 districts and their count of affected people.', pad=20,fontsize=40)
plt.xlabel('Count of people affected', fontsize=30)
plt.ylabel('District', fontsize=30)
# plt.rc('xtick',labelsize=0)
# plt.rc('ytick',labelsize=20)
plt.show()


# > Graph showing relationship between current status and age of people

# In[ ]:


plt.figure(figsize=(15,10))
ax=sns.swarmplot(data['current_status'],data['age'],palette=sns.dark_palette("blue", reverse=True))
ax.set_title('Graph between age and current status of affected people', pad=20,fontsize=20)
plt.xlabel('Current Status', fontsize=20)
plt.ylabel('Age', fontsize=20)
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)


# #### This swarmplot shows that majority of the people hospitalized are in the age group of ~18 to 88

# #### No. of days between change in status

# In[ ]:



data.head()


# #### Days taken to change status(Hospitalized/Death/Recover)

# In[ ]:


data['status change difference'] = pd.to_datetime(data['status_change_date'])-pd.to_datetime(data['diagnosed_date'])


# In[ ]:


data['status change difference'].value_counts()


# ### Graph to show relationship between status change

# ---

# In[ ]:


data.head()


# In[ ]:


diagnose_data=pd.DataFrame({'Dates':pd.to_datetime(data['diagnosed_date']).value_counts().index,
                           'Count':pd.to_datetime(data['diagnosed_date']).value_counts().values})
diagnose_data=diagnose_data.sort_values('Dates')
diagnose_data=diagnose_data.reset_index(drop=True)
diagnose_data=diagnose_data.set_index('Dates')


# In[ ]:


import matplotlib.dates as mdates
fig, ax = plt.subplots(figsize=(20,10))
graph=ax.plot(diagnose_data.Count,marker='s')
myFmt = mdates.DateFormatter('%d-%m-%y')
ax.xaxis.set_major_formatter(myFmt)
plt.rc('xtick',labelsize=8)
plt.xticks(diagnose_data.index.values,rotation=90)
plt.show()


# In[ ]:


diagnose_data['cumsum']=diagnose_data['Count'].cumsum()


# In[ ]:


diagnose_data['cumsum']


# In[ ]:


fig, ax = plt.subplots(figsize=(20,10))
graph=ax.plot(diagnose_data['cumsum'],marker='s')
myFmt = mdates.DateFormatter('%d-%m-%y')
ax.xaxis.set_major_formatter(myFmt)
plt.rc('xtick',labelsize=8)
plt.xticks(diagnose_data.index.values,rotation=90)
plt.show()


# ### Data fetched from local news about the shutdown process in India.

# In[ ]:


entities = ['schools & other educational organization shutdown','public places shutdown',
            'work from home started for employees','country under lockdown','after lockdown']
dates = ['2020-03-10','2020-03-13','2020-03-16','2020-03-21','2020-03-26']


# Using  boolean mask to fetch case counts within dates

# In[ ]:


# Now getting the cases count before each entity shutdown.
data = data.sort_values(by="diagnosed_date")


# In[ ]:


mask1 = (data['diagnosed_date'] > '2020-01-30') & (data['diagnosed_date'] <= '2020-03-09') # before school lockdown
mask1 = len(data.loc[mask1])
mask2 = (data['diagnosed_date'] > '2020-03-10') & (data['diagnosed_date'] <= '2020-03-12') #before public place shutdown
mask2 = len(data.loc[mask2])
mask3 = (data['diagnosed_date'] > '2020-03-13') & (data['diagnosed_date'] <= '2020-03-15') # beforw wfh
mask3 = len(data.loc[mask3])
mask4 = (data['diagnosed_date'] > '2020-03-16') & (data['diagnosed_date'] <= '2020-03-21') # beforw lookdown
mask4 = len(data.loc[mask4])
mask5 = (data['diagnosed_date'] > '2020-03-22') & (data['diagnosed_date'] <= '2020-03-28') # beforw lookdown
mask5 = len(data.loc[mask5])


# In[ ]:


case_counts = [mask1, mask2,mask3,mask4,mask5]


# In[ ]:


plot_data = pd.DataFrame({'dates':dates,'counts':case_counts,'entities': entities})


# In[ ]:


plot_data['dates'] = pd.to_datetime(plot_data['dates'])


# In[ ]:


plot_data


# In[ ]:


fig, ax = plt.subplots(figsize=(10,8))
plt.hlines(y=plot_data.entities, xmin=0, xmax=plot_data.counts, color='red')
plt.plot(plot_data.counts, plot_data.entities, "D")
# Add titles and axis names
ax.xaxis.label.set_color('black')
plt.yticks(plot_data.entities)
plt.title("Case Count Growth Rate After Lockdown")
plt.xlabel('Covid19 Case Count', fontsize=10)


# #### Analysis: Even after lockdown more cases are being reported everyday

# In[ ]:


def get_travel_hitsory(text):    
    doc = ner.name(text, language='en_core_web_sm')
    text_label = set((X.text,X.label_) for X in doc)
    if  not text_label:
        return "Unknown"
    d = dict(list(text_label))
    for i in d:
        if d[i] == 'GPE':
            return i
        else:
            return "Unknown"


# In[ ]:


data['travel_from']=data['notes'].apply(get_travel_hitsory)


# In[ ]:


data['detected_city_latlong'] = data['detected_city_pt'].apply(get_lat_long)

data['detected_city_lat'] = data['detected_city_latlong'].apply(lambda x:x[0])
data['detected_city_long'] = data['detected_city_latlong'].apply(lambda x:x[1])

data.detected_city_lat = data.detected_city_lat.astype('float64')
data.detected_city_long = data.detected_city_long.astype('float64')


# In[ ]:


network_data = pd.DataFrame({'travel_from':data.travel_from,'travel_to_lat':data.detected_city_lat,'travel_to_long':data.detected_city_long})


# In[ ]:


network_data


# In[ ]:


indexNames = network_data[network_data['travel_from'] == "Unknown" ].index
# Delete these row indexes from dataFrame
network_data.drop(indexNames , inplace=True)
indexNames = network_data[network_data['travel_from'] == "Arrived" ].index
# Delete these row indexes from dataFrame
network_data.drop(indexNames , inplace=True)


# In[ ]:


network_data['travel_from'].value_counts()


# In[ ]:


plt.figure(figsize=(25,20))
ax=sns.barplot(network_data['travel_from'].value_counts()[:10].values,network_data['travel_from'].value_counts()[:10].index, palette=sns.dark_palette("blue", reverse=True))
for i in ax.patches:
    ax.text(i.get_width()+0.50, i.get_y()+0.50,             str(int(i.get_width())), fontsize=30,color='black')
ax.set_title('Top 10 places from where infected people had travelled.', pad=20,fontsize=40)
plt.xlabel('Count of people ', fontsize=30)
plt.ylabel('Places', fontsize=30)
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.show()


# In[ ]:




