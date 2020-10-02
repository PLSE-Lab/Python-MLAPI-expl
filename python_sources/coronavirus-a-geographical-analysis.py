#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# import the necessary libraries
import numpy as np 
import pandas as pd 

# Visualisation libraries
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
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

# Disable warnings 
import warnings
warnings.filterwarnings('ignore')


# ### Update (4th Feb) : 
# * The code in the notebook has been updated to generalise for all dates and countries.
# * Also, I am making the [world_coordinates.csv](https://www.kaggle.com/parulpandey/world-coordinates) file available which contains the latitudes and longitudes of all the countries of the world. The dataset has been scraped from the net.
#                  

# # Wuhan Coronavirus : A geographical analysis
# 
# With the news coming in that the [World Health Organization has declared the novel coronavirus outbreak a public health emergency](https://edition.cnn.com/2020/01/30/health/coronavirus-who-public-health-emergency-international-concern-declaration/index.html), it has increased the general fear among the public. A lot of countires have heightened their measures to fight with this virus with the condition in China still senitive..More than 20 countries and territories outside of mainland China have confirmed cases of the virus -- spanning Asia, Europe, North America and the Middle East -- as India, Italy and the Philippines reported their first cases on Thursday.[source](https://edition.cnn.com/2020/01/30/asia/wuhan-coronavirus-update-intl-hnk/index.html)

# In[ ]:


#from IPython.display import IFrame
#IFrame('https://www.myheatmap.com/maps/PPk1_rfT1jQ%3D', width=800, height=600)


# ![](https://imgur.com/q1JolVI.png)

# In[ ]:


# Reading the dataset
corona_virus=("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")
data= pd.read_csv(corona_virus)
data.head()


# In[ ]:


# Let's look at the various columns
data.info()


# In[ ]:


# Convert Last Update column to datetime64 format

data['Date'] = data['Date'].apply(pd.to_datetime)
data.drop(['Sno'],axis=1,inplace=True)

#Set Date column as the index column.
#data.set_index('Last Update', inplace=True)
data.head()


# ## Countries which have been affected by the Coronavirus(2019-nCoV)till now

# In[ ]:


# Countries affected

countries = data['Country'].unique().tolist()
print(countries)

print("\nTotal countries affected by virus: ",len(countries))


# China and Mainland China have been reported separately. Total 33 countries are been affected by the Coronavirus.

# In[ ]:


#Combining China and Mainland China cases

data['Country'].replace({'Mainland China':'China'},inplace=True)
countries = data['Country'].unique().tolist()
print(countries)
print("\nTotal countries affected by virus: ",len(countries))


# In total 32 countries are affected by combining the China and Mainland China.
# Just to clear whether the virus is contagious or not. It can be seen that more and more cases are been reported and are more than what was confirmed in the begining. It is for sure from the Data that the virus is transmissible.

# In[ ]:


d = data['Date'][-1:].astype('str')
year = int(d.values[0].split('-')[0])
month = int(d.values[0].split('-')[1])
day = int(d.values[0].split('-')[2].split()[0])

from datetime import date
data_latest = data[data['Date'] > pd.Timestamp(date(year,month,day))]
data_latest.head()


# In[ ]:


# Creating a dataframe with total no of confirmed cases for every country
Number_of_countries = len(data_latest['Country'].value_counts())


cases = pd.DataFrame(data_latest.groupby('Country')['Confirmed'].sum())
cases['Country'] = cases.index
cases.index=np.arange(1,Number_of_countries+1)

global_cases = cases[['Country','Confirmed']]
#global_cases.sort_values(by=['Confirmed'],ascending=False)
global_cases


# It can be seen that highest confirmed cases are from the China, from where the virus actually spreaded.

# Let's create a new dataframe which consists of the latitude and longitudes of the countries under observation. It will help to understand that which areas/provinces in the countries are affected by the virus.

# In[ ]:


# Importing the world_coordinates dataset
world_coordinates = pd.read_csv('../input/world-coordinates/world_coordinates.csv')

# Merging the coordinates dataframe with original dataframe
world_data = pd.merge(world_coordinates,global_cases,on='Country')
world_data.head()


# ## Visualizing the current world scenario

# In[ ]:



# create map and display it
world_map = folium.Map(location=[10, -20], zoom_start=2.3,tiles='Stamen Toner')

for lat, lon, value, name in zip(world_data['latitude'], world_data['longitude'], world_data['Confirmed'], world_data['Country']):
    folium.CircleMarker([lat, lon],
                        radius=10,
                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Confirmed Cases</strong>: ' + str(value) + '<br>'),
                        color='red',
                        
                        fill_color='red',
                        fill_opacity=0.7 ).add_to(world_map)
world_map


# Though the latitude and longitude confirms the countries in which the virus is spread, but exact locastion is untrackable. For example, There are 3 confirmed cases in Kerala India. But as per the latitude and longitude, its pointing the Maharashtra province.

# In[ ]:


# A look at the different cases - confirmed, death and recovered
print('Globally Confirmed Cases: ',data_latest['Confirmed'].sum())
print('Global Deaths: ',data_latest['Deaths'].sum())
print('Globally Recovered Cases: ',data_latest['Recovered'].sum())


# From the Death and Recovered data, it can be seen that recovered to death ratio is 4:1. It means that the disease is recoverable. Recovery also depends on the stage at which patient is treated i.e either in early stage or severe stage. Unfortunately, It cannot be analysis because of the data limitation.

# In[ ]:


# Let's look the various Provinces/States affected

data_latest.groupby(['Country','Province/State']).sum()


# In[ ]:


# Provinces where deaths have taken place
data_latest.groupby('Country')['Deaths'].sum().sort_values(ascending=False)


# Apart from China,Philippines and Hong kong, there is no death in other countries.

# In[ ]:



# Lets also look at the Recovered stats
data_latest.groupby('Country')['Recovered'].sum().sort_values(ascending=False)[:5]


# ## A Closer look at China's condition

# In[ ]:


#Mainland China
China = data_latest[data_latest['Country']=='China']
China


# ### Let's look at the Confirmed vs Recovered figures of Provinces of China other than Hubei

# In[ ]:


f, ax = plt.subplots(figsize=(12, 8))

sns.set_color_codes("pastel")
sns.barplot(x="Confirmed", y="Province/State", data=China[1:],
            label="Confirmed", color="r")

sns.set_color_codes("muted")
sns.barplot(x="Recovered", y="Province/State", data=China[1:],
            label="Recovered", color="g")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 400), ylabel="",
       xlabel="Stats")
sns.despine(left=True, bottom=True)


# Its really interesting to see that though it is been said that the virus has no cure, there has been recovery.
# More information can help to know the avarage recovery time,at which stage the patient has a chances to get recover.
# 
# 
# 
