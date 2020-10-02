#!/usr/bin/env python
# coding: utf-8

# ### Breakdown of this notebook:
# 
# - **Loading the dataset:** Load the data and import the libraries.
# - The data is already almost clean without any missing values,etc.
# - **Dividing** the data into **decades**
# - Analysing and visualizing **immigration per continent** with every decade.
# - Finding the **top countries** with most immigrants over each decade.
# - **Developing** vs **Developed** countries
# - **Choropleth map** for visualizing immigrants to canada from across the world.

# References:-
# Thanks for sharing the choropleth maps Roshan!
# https://www.kaggle.com/roshansharma/immigration-to-canada

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


df = pd.read_excel('../input/Canada.xlsx',
                     sheet_name='Canada by Citizenship',
                     skiprows = range(20),
                     skipfooter = 2)

# getting the shape of the data
df.shape


# In[3]:


df.head(5)


# #### There are no null or NA values in the dataset as you can see below. Wish the data was dirtier !! 

# In[4]:


print("The number of nulls in each column are \n", df.isna().sum())


# In[5]:


df.iloc[2].nunique()


# ### Dropping the redundant columns 

# The columns **Area , REG ,DEV ,Type, Coverage** don't seem to have any information that can be helpful in our problem.

# In[6]:


df = df.drop(['AREA','REG','DEV','Type','Coverage'], axis = 1)

# adding a Total column to add more information
df['Total'] = df.sum(axis = 1)

# let's check the head of the cleaned data
df.head()


# In[7]:


df['decade1']=df.iloc[:,4:14].sum(axis=1)
df['decade2']=df.iloc[:,14:24].sum(axis=1)
df['decade3']=df.iloc[:,24:34].sum(axis=1)
df['decade4']=df.iloc[:,34:38].sum(axis=1)
df.head()


# ### Dividing the year data into decades.
# 
# - decade1= 1980-1989
# - decade2= 1990-1999
# - decade3= 2000-2009
# - decade4= 2010-2013
# 
# This will help us understand the inflow of immigrants from across the world to Canada with each passing decade.   
# *Note that decade4 only consists of 3 years of data, but we can still draw some conclusions from it.*

# In[8]:


df1=df[['AreaName','decade1']].groupby(['AreaName']).sum(axis=1).sum(
    level=['AreaName'])
df2=df[['AreaName','decade2']].groupby(['AreaName']).sum(axis=1).sum(
    level=['AreaName'])
df3=df[['AreaName','decade3']].groupby(['AreaName']).sum(axis=1).sum(
    level=['AreaName'])
df4=df[['AreaName','decade4']].groupby(['AreaName']).sum(axis=1).sum(
    level=['AreaName'])
new_df=pd.merge(df1, df2, how='inner', on = 'AreaName')
new_df=pd.merge(new_df, df3, how='inner', on = 'AreaName')
new_df=pd.merge(new_df, df4, how='inner', on = 'AreaName')
print("Total number of immigrants per decade \n ",new_df)


# ### Plotting the immigrants per continent

# Now, from the above obtained dataframe, we will plot immigrants from **each continent** to Canada for each decade. 

# In[9]:


ax2=new_df.plot(kind = 'bar', color=['red', 'green', 'blue','yellow'], figsize = (15,6), rot = 70)
labels = ['decade1','decade2','decade3','decade4']
ax2.legend(labels = labels)
ax2.set_xlabel('Immigration by continents')
ax2.set_ylabel('Immigrants')
plt.show()


# * For the **first decade(1980-1989)**, the number of immigrants from **Europe** and **Asia** were almost the **same**.
# * For the **second decade(1990-1999)**, we see a **sudden upsurge in immigrants from Asia and Europe**. Immigrants from Latin America and Carribean also increase a little.
# * For the **third decade(2000-2009)** , interestingly we see that number of immigrants from **Asia almost double from the previous decade**. Contrastingly, **Europe, Latin America, Northern America and Oceania** show a **decrease** in the immigrants from previous decade.
# * For the **fourth incomplete decade(2010-2013)**, we see that **Asia** has already seen half of last decade's number in just 3 years.

# ### Stacked representation of Immigrants per continent

# In[10]:


new_df.plot(kind="bar", 
                 figsize=(8,8),
                 stacked=True)


# ### Analyzing data by countries

# **Let us see which countries individually contributed the most to the number of immigrants to Canada.**

# In[11]:


def create_plot(newc,decade):
    newc.plot(kind="bar",figsize=(10, 10))
    plt.ylabel('Number of immigrants')
    plt.xlabel('Countries')
    plt.title('Immigrant distribution for '+decade)


# In[12]:


decades = ['decade1','decade2','decade3','decade4']
plt.figure(1,figsize=(10, 10))
for decade in decades:
    country=df[['OdName',decade]].groupby(['OdName']).sum(axis=1).sum(level=['OdName'])
    # print(country)
    mean=country.mean()
    newc = country[(country > mean).all(axis=1)]
#     print (newc)
#     indices= decades.index(decade)
#     print(indices+1)
#     plt.subplot(4,1,indices+1)
#     plt.subplots_adjust(hspace=0.9)
    create_plot(newc,decade)


# #### Countries in order of largest immigrants:-
# - **Great Britain(largest contributor)**, USA, India, Philippines and China for the **first decade**.
# - **Great Britain(largest contributor)**, India, Philippines and Poland for **second decade**.
# - **China(largest contributor)**,India, Great Britain, Pakistan, Philippines for **third decade**.
# - **China(largest contributor)**,India, Pakistan, Philippines for the **fourth decade**.
# 
# *Its interesting how the numbers have completely changed for* **China** *and* **Great Britain** *over the years!!*

# 
# 
# **Now,lets see the country background of all the immigrants over the years**
# 
# 

# In[13]:


plt.style.use('_classic_test')

colors = plt.cm.cool(np.linspace(0, 50, 100))
df['DevName'].value_counts().plot.pie(colors = colors,
                                       figsize = (10, 10))

plt.title('Types of Countries', fontsize = 20, fontweight = 30)
plt.axis('off')
plt.legend()
plt.show()


# - The **developed countries** take almost **25%** of all countries who immigrated to Canada over all the years.
# - **This is misleading !** *This chart does not show the total population share of developed countries. It merely shows that of all countries who migrated to Canada , 75% are developing and 25% are developed.*

# In[14]:


df1=df[['DevName','decade1']].groupby(['DevName']).sum(axis=1).sum(
    level=['DevName'])
df2=df[['DevName','decade2']].groupby(['DevName']).sum(axis=1).sum(
    level=['DevName'])
df3=df[['DevName','decade3']].groupby(['DevName']).sum(axis=1).sum(
    level=['DevName'])
df4=df[['DevName','decade4']].groupby(['DevName']).sum(axis=1).sum(
    level=['DevName'])
new_df=pd.merge(df1, df2, how='inner', on = 'DevName')
new_df=pd.merge(new_df, df3, how='inner', on = 'DevName')
new_df=pd.merge(new_df, df4, how='inner', on = 'DevName')
print("Background of immigrants per decade \n ",new_df)


# In[15]:


def create_pi_plot(new_df,decade):
    new_df.plot(kind="bar", 
                 figsize=(8,8),
                 stacked=True)


# In[16]:


plt.figure(1,figsize=(10, 10))
create_pi_plot(new_df,decades)


# #### This graph correctly shows the immigrant population share of developing vs developed countries to Canada.
# 
# - Immigrants from developed and developing countries are **comparable** in **first decade**.
# - The ratio between Developing and Developed immigrants keeps on almost **doubling** every decade after first decade.

# ### Lastly, we should visualise the total immigrants over all the years on the world map.

# In[17]:


# download countries geojson file
import folium
# download countries geojson file
get_ipython().system('wget --quiet https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/world_countries.json -O world_countries.json')
    
print('GeoJSON file downloaded!')
world_geo = r'world_countries.json' # geojson file

# create a plain world map
world_map = folium.Map(location=[0, 0], zoom_start=2, tiles='Mapbox Bright')
import warnings
warnings.filterwarnings('ignore')

# generate choropleth map using the total immigration of each country to Canada from 1980 to 2013
world_map.choropleth(
    geo_data=world_geo,
    data=df,
    columns=['OdName', 'Total'],
    key_on='feature.properties.name',
    fill_color='Greens', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Immigration to Canada'
)

# display map
world_map


# ### Please upvote and feel free to give any feedback/comment below!!
