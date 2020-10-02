#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import pycountry
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#read data
earthquake_data = pd.read_csv('/kaggle/input/global-significant-earthquake-database-from-2150bc/Worldwide-Earthquake-database.csv',index_col=0)
earthquake_data.head() #Preview data


# In[ ]:


earthquake_data.info()


# In[ ]:


earthquake_data.describe()


# In[ ]:


earthquake_data.isnull().sum()


# In[ ]:


earthquake_data.shape #Total number of row and columns 


# As we can see total number of data (Rows) is 6193. Almost all attribute are having more tham 50% of null value. If we remove all missing value, we will end up with no data. So I have decided to leave null value as it is.

# In[ ]:



data_year= earthquake_data.groupby("YEAR")["FOCAL_DEPTH"].max()

data_year.sort_values(ascending=False).head(10).plot(kind="barh",title="Which year has maximum depthed earthquake?",color='coral', figsize=(10,6))
plt.xlabel("Focal depth")


# From above graph, it is cleared in **2002 the earthquake was having maximum value of focal deapth.**

# In[ ]:


from_1900to2000=earthquake_data[earthquake_data['YEAR']>=1900] #filter data 

fig = plt.figure(figsize=(16,8)) 
country = from_1900to2000.groupby("COUNTRY")["YEAR"].count()
country.sort_values(ascending=False).head(20).plot(kind="bar",color='tan', title="Which country has faced more number of earthquake from 1900 to 2020?")
plt.ylabel("Total number of earthquake")


# Above data is only for top 20 country. I have first filter the data from 1900 to 2020 as the original data was containing negative values which was misspelled. **The China has faced highest number of earth quake** more tha 350 times from 1900 to 2020.Indonesia is legging China by 100 counts.

# In[ ]:


fig = plt.figure(figsize=(16,8)) 
colors_list = ['skyblue', 'yellowgreen']
explode_list = [0,0.05]
earthquake_data['T']= np.where(earthquake_data['FLAG_TSUNAMI']=='Yes', 1, 0)
eq=earthquake_data['T'].value_counts()
eq.plot(kind='pie',
        figsize=(15, 6),
        autopct='%1.1f%%', 
        startangle=90,    
        shadow=True,       
        colors=colors_list,
        pctdistance=0.5,# add custom colors
        explode=explode_list, 
        labels=["No Tsunami due to earthquak","Tsunami due to earthquake"]
        )
plt.title("Tsunami due to earthquake") 
plt.legend(labels=["No","Yes"], loc='upper left')


# Due to earthquake almost 30% Tsunami occured.

# In[ ]:


is_tsunami=from_1900to2000[from_1900to2000['FLAG_TSUNAMI']=='Yes']
fig = plt.figure(figsize=(16,8)) 
country = is_tsunami.groupby("COUNTRY")["YEAR"].count()
country.sort_values(ascending=False).head(20).plot(kind="bar",color='limegreen', title="Which country has faced more number of Tsunami from 1900 to 2020?")
plt.ylabel("Total number of Tsunami")


# Japan has faced highest number of Tsunami from 1900 to 2020.

# In[ ]:


total_death=earthquake_data.groupby('COUNTRY')['DEATHS'].sum().reset_index(name ='D_COUNT')
countries = {}
for country in pycountry.countries:
    countries[country.name.upper()] = country.alpha_3

total_death['code'] = [countries.get(country, np.NaN) for country in total_death['COUNTRY']]

print(total_death[total_death['code'].isnull()])


# 

# <p>The goal is to find out total death and plot that data on world map. Hence, I grouped by country.<br> To plot data on world map I required iso3 code of country. For this I used python library pycountry. But due to mismatch in country name of library and dataset, total 31 code came null. There are some country having high number of death count were having code valur null.

# In[ ]:


code_dics={'AZORES (PORTUGAL)':'PRT','BOLIVIA':'BOL','BOSNIA-HERZEGOVINA':'BIH',
       'IRAN':'IRN','KERMADEC ISLANDS (NEW ZEALAND)':'NZL','MACEDONIA':'MKD',
       'MYANMAR (BURMA)':'MMR','NORTH KOREA':'PRK','RUSSIA':'RUS','SOUTH KOREA':'KOR',
       'SYRIA':'SYR','TAIWAN':'TWN','UK':'GBR','USA':'USA','VENEZUELA':'VEN', 'TANZANIA':'TZA', 'VIETNAM':'VNM',
        'CZECH REPUBLIC':'CZE'}
print(total_death.columns)
for key,value in code_dics.items():
    total_death.loc[total_death['COUNTRY'].eq(key), 'code'] = value


# To change code value from null I created dictionary. 

# In[ ]:


total_death.dropna(inplace=True)


# In[ ]:


import plotly.express as px
import geopandas as gpd
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

for_plotting = world.merge(total_death,left_on = 'iso_a3', right_on = 'code')

ax = for_plotting.dropna().plot(column='D_COUNT', cmap =    
                                'YlGnBu', figsize=(15,9),   
                                 scheme='quantiles', legend =  
                                  True);
ax.set_title("Earthquake fatalities")
plt.show()


# In[ ]:




