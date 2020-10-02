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
print(os.listdir("../input"))
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


['indiarain', 'keraladistricts', 'rainfall-in-india']


# **Importing modules needed for carrying out out work**

# In[ ]:


import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns


# **Importing map of rainfall distribution across India**

# In[ ]:


img=np.array(Image.open('../input/annualmeanrainfallmapofindia/Annual-mean-rainfall-map-of-India.png'))
fig=plt.figure(figsize=(10,10))
plt.imshow(img,interpolation='bilinear')
plt.axis('off')
plt.ioff()
plt.show()


# From the above picture we can see that India receives heavy rainfall in coastal South West region and in the North East.The Shayadri and the Himalayan mountain ranges obstruct the clouds which cause heavy rainfall in these regions.East and Central India receive moderate annual rainfall.North West India which includes Thar desert receives scanty annual ranfall.

# **Inporting the 115 years of Indian rainfall data**

# In[ ]:


India = pd.read_csv("../input/rainfall-in-india/rainfall in india 1901-2015.csv",sep=",")


# **Displaying the head of the data**

# In[ ]:


India.head()


# We have 115 years of monthly,seasonal and annual rainfall data of India in the dataset

# **Summary of Dataset**

# In[ ]:


print('Rows     :',India.shape[0])
print('Columns  :',India.shape[1])
print('\nFeatures :\n     :',India.columns.tolist())
print('\nMissing values    :',India.isnull().values.sum())
print('\nUnique values :  \n',India.nunique())


# **Inspecting the data**

# In[ ]:


total = India.isnull().sum().sort_values(ascending=False)
percent = (India.isnull().sum()/India.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
f, ax = plt.subplots(figsize=(15, 6))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data['Percent'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
missing_data.head()


# We can clearly See that the Annual Rainfall column has maximum missing values.We have to divise a strategy to fill up this missing values

# We can see that there are some missing values in the data set.

# **Type of values in the data set**

# In[ ]:


India.info()


# **Replacing the missing values with mean of the column**

# In[ ]:


India['JAN'].fillna((India['JAN'].mean()), inplace=True)
India['FEB'].fillna((India['FEB'].mean()), inplace=True)
India['MAR'].fillna((India['MAR'].mean()), inplace=True)
India['APR'].fillna((India['APR'].mean()), inplace=True)
India['MAY'].fillna((India['MAY'].mean()), inplace=True)
India['JUN'].fillna((India['JUN'].mean()), inplace=True)
India['JUL'].fillna((India['JUL'].mean()), inplace=True)
India['AUG'].fillna((India['AUG'].mean()), inplace=True)
India['SEP'].fillna((India['SEP'].mean()), inplace=True)
India['OCT'].fillna((India['OCT'].mean()), inplace=True)
India['NOV'].fillna((India['NOV'].mean()), inplace=True)
India['DEC'].fillna((India['DEC'].mean()), inplace=True)
India['ANNUAL'].fillna((India['ANNUAL'].mean()), inplace=True)
India['Jan-Feb'].fillna((India['Jan-Feb'].mean()), inplace=True)
India['Mar-May'].fillna((India['Mar-May'].mean()), inplace=True)
India['Jun-Sep'].fillna((India['Jun-Sep'].mean()), inplace=True)
India['Oct-Dec'].fillna((India['Oct-Dec'].mean()), inplace=True)


# In[ ]:


India.describe().T


# **Annual rainfall in India**

# In[ ]:


ax=India.groupby("YEAR").mean()['ANNUAL'].plot(ylim=(600,2200),color='b',marker='o',linestyle='-',linewidth=2,figsize=(12,8));
India['MA10'] = India.groupby('YEAR').mean()['ANNUAL'].rolling(10).mean()
India.MA10.plot(color='r',linewidth=4)
plt.xlabel('Year',fontsize=20)
plt.ylabel('Annual Rainfall (in mm)',fontsize=20)
plt.title('Annual Rainfall in India from Year 1901 to 2015',fontsize=25)
ax.tick_params(labelsize=15)
plt.grid()
plt.ioff()


# Highest average rainfall in India was recored in the year 1961.This was because in 1961 India received multiple cyclones.City of Pune was flooded in the year 1961 which is remembered as Panshet Flood.Due to heavy rains Panshet dam developed cracks and gave way.While the dam started cracking Indian army carried out emergency operation to stop flow of water.Sand bags were used to stop the flow of water.It gave people some time to evacuate before the dam was destroyed.Early morning water entered the city had caused massive damage.
# 
# Year 1965-66 were twin drought years and there was food scarcity in India.Prime Minister Lal Bahadur Shastri gave the Slogan Jai Jawan Jai Kissan to people of India.This lead to green revolution in India making India a food surplus country in the coming decades.
# 
# The red line is the 10 year moving average of the rainfall in India.It seems since 1960s there is slight dip in the rainfall in India.Now a days due to global warming the period of Monsoon season has shortned.We see more of erratic rainfall pattern.This needs more study.

# **Seasonal rainfall in India**

# In[ ]:


India[['YEAR','Jan-Feb', 'Mar-May',
       'Jun-Sep', 'Oct-Dec']].groupby("YEAR").mean().plot(figsize=(13,8));
plt.xlabel('Year',fontsize=20)
plt.ylabel('Seasonal Rainfall (in mm)',fontsize=20)
plt.title('Seasonal Rainfall from Year 1901 to 2015',fontsize=25)
ax.tick_params(labelsize=15)
plt.grid()
plt.ioff()


# Plotting a seasonal rainfall we can see that highest rainfall is received between Jun-Sep which is the monsoon season in India.Oct-Dec is the season of return monsoon and cyclone season in the Bay of Bengal.A major city like Chennai only receives the return monsoon.Jan-Feb receives very less rainfall as this is the winter season across the sub-continent.Mar-May is the time for summer rains which is generally acompanied by thunder storms.

# **Season wise rainfall in India**

# In[ ]:


India[['SUBDIVISION', 'Jan-Feb', 'Mar-May',
       'Jun-Sep', 'Oct-Dec']].groupby("SUBDIVISION").mean().sort_values('Jun-Sep').plot.bar(width=0.5,edgecolor='k',align='center',stacked=True,figsize=(16,8));
plt.xlabel('Subdivision',fontsize=30)
plt.ylabel('Rainfall (in mm)',fontsize=20)
plt.title('Rainfall in Subdivisions of India',fontsize=25)
ax.tick_params(labelsize=15)
plt.grid()
plt.ioff()


# In[ ]:


drop_col = ['ANNUAL','Jan-Feb','Mar-May','Jun-Sep','Oct-Dec']

fig, ax = plt.subplots()

(India.groupby(by='YEAR')
 .mean()
 .drop(drop_col, axis=1)
 .T
 .plot(alpha=0.1, figsize=(12, 6), legend=False, fontsize=12, ax=ax)
)
ax.set_xlabel('Months', fontsize=12)
ax.set_ylabel('Rainfall (in mm)', fontsize=12)
plt.grid()
plt.ioff()


# 1.From the above graph we can see that majority of rainfall is received in the month of Jun-Sep which is the Monsoon season.Oct-Dec is time of return monsoon.Jan-Feb are the winter months.Mar-May is time for Summer rains.
# 
# 2.Coastal Karnataka,Arunachal Pradesh,Konkan Goa and Kerala receive highest rainfall.
# 
# 3.Rajastan,Gujrat,Haryana and Punjab receives low rainfall.Interesting thing is that Punjab and Haryana have high agricultural output despite low rainfall.Their water requirnments are met by rivers and canals.

# **Box Plot of Annual Rainfall**

# In[ ]:


plt.rcParams['figure.figsize']=(23,10)
ax = sns.boxplot(x="SUBDIVISION", y="ANNUAL", data=India,width=0.8,linewidth=3)
ax.set_xlabel('Subdivision',fontsize=30)
ax.set_ylabel('Annual Rainfall (in mm)',fontsize=30)
plt.title('Annual Rainfall in Subdivisions of India',fontsize=40)
ax.tick_params(axis='x',labelsize=20,rotation=90)
ax.tick_params(axis='y',labelsize=20,rotation=0)
plt.grid()
plt.ioff()


# We can see Subdivision Arunachal Pradesh shows highest highest difference between Maximum and Minimum rainfall received.Costal Karnataka receives close to 3400 mm of Annual rainfall which is the highest in India.West Rajastan receives the least amount of rainfall.

# In[ ]:


fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
India.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL', ascending=False)['ANNUAL'].plot('bar', color='b',width=0.65,linewidth=3,edgecolor='k',align='center',title='Subdivision wise Average Annual Rainfall', fontsize=20)
plt.xticks(rotation = 90)
plt.ylabel('Average Annual Rainfall (in mm)')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
#print(India.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL', ascending=False)['ANNUAL'][[0,1,2]])
#print(India.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL', ascending=False)['ANNUAL'][[33,34,35]])


# **Monthwise Rainfall in India**

# In[ ]:


ax=India[['JAN', 'FEB', 'MAR', 'APR','MAY', 'JUN', 'AUG', 'SEP', 'OCT','NOV','DEC']].mean().plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2,figsize=(16,8))
plt.xlabel('Month',fontsize=30)
plt.ylabel('Monthly Rainfall (in mm)',fontsize=20)
plt.title('Monthly Rainfall in India',fontsize=25)
ax.tick_params(labelsize=20)
plt.grid()
plt.ioff()


# In[ ]:


India[['AUG']].mean()


# We can see maximum rainfall of 290 mm is received in the month of August which is the peak of Monsoon Season.

# **Heat Map of Rainfall**

# In[ ]:


#India1=India['JAN','FEB','ANNUAL']
fig=plt.gcf()
fig.set_size_inches(15,15)
fig=sns.heatmap(India.corr(),annot=True,cmap='summer',linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)


# We can see that Annual rainfall has very high correlation to the rainfall received in the months of Jun-Sep

# **Kerala Rainfall**

# In[ ]:


img=np.array(Image.open('../input/kerela-district-map/Kerala-district-Map.png'))
fig=plt.figure(figsize=(10,10))
plt.imshow(img,interpolation='bilinear')
plt.axis('off')
plt.ioff()
plt.show()


# Kerala is Southern coastal state of India.It is known as Gods Own Country for its scenic beauty.One one side it is surrounded by the Arabian sea and on the other side by the Shayadari mountain range.Many tourists visit state of Kerala for undergoing Ayurvedic healing ,for its sandy beaches,greenery and backwaters.The state of Kerala has 14 districts as shown in the above map.

# In[ ]:


Kerala =India[India.SUBDIVISION == 'KERALA']
#Kerala


# In[ ]:


ax=Kerala.groupby("YEAR").mean()['ANNUAL'].plot(ylim=(1000,5000),color='b',marker='o',linestyle='-',linewidth=2,figsize=(12,8));
#Kerala['MA10'] = Kerala.groupby('YEAR').mean()['ANNUAL'].rolling(10).mean()
#Kerala.MA10.plot(color='r',linewidth=4)
plt.xlabel('Year',fontsize=20)
plt.ylabel('Kerala Annual Rainfall (in mm)',fontsize=20)
plt.title('Kerala Annual Rainfall from Year 1901 to 2015',fontsize=25)
ax.tick_params(labelsize=15)
plt.grid()
plt.ioff()


# In[ ]:


print('Average annual rainfall received by Kerala=',int(Kerala['ANNUAL'].mean()),'mm')


# In[ ]:


print('Kerala received 4257.8 mm of rain in the year 1961')
a=Kerala[Kerala['YEAR']==1961]
a


# In[ ]:


print('Kerala received 4226.4 mm of rain in the year 1924')
b=Kerala[Kerala['YEAR']==1924]
b


# Prior to 2018 Kerala had major flood in the year 1924 which is evident in the data.Contrary to popular belief Kerala received maximum annual rainfall in year 1961(4257 mm) and not 1924(4226 mm).In 2018 Kerala has received 2226.4 mm of rain in the monsoon season.This is 40% more than the average rainfall.We have to wait till the end of the year to see if monsoon will break the 1961 rainfall record in Kerala.

# **Importing the Districtwise rainfall data**

# In[ ]:


Dist = pd.read_csv("../input/rainfall-in-india/district wise rainfall normal.csv",sep=",")


# In[ ]:


Dist.head()


# **Annual rainfall in different districts of Kerala**

# In[ ]:


KDist=Dist[Dist.STATE_UT_NAME == 'KERALA']
k=KDist.sort_values(by=['ANNUAL'])
ax=k.plot.bar(x='DISTRICT',y='ANNUAL',width=0.5,edgecolor='k',align='center',linewidth=4,figsize=(16,8))
plt.xlabel('District',fontsize=30)
plt.ylabel('Annual Rainfall (in mm)',fontsize=20)
plt.title('Rainfall in Districts of Kerala',fontsize=25)
ax.tick_params(labelsize=20)
plt.grid()
plt.ioff()


# From the graph we can see that Kasargod which is Northern most district of Kerala receives highest annual rainfall.Trivandrum which is the southern most district receives the least amount of rainfall.From this we can clearly make out that rainfall increases as we more from South to North in the State of Kerala.This could be due to proximity of Shayadri mountain ranges in the Northern disctrict of Kerala.

# **Five districts with least rainfall in India**

# In[ ]:


Dist.groupby(['DISTRICT'])['ANNUAL'].max().sort_values().head(5)


# **Five districts with maximum rainfall in India**

# In[ ]:


ax=Dist.groupby(['DISTRICT'])['ANNUAL'].max().sort_values().head(20).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=4,figsize=(16,8))
#ax=k.plot.bar(x='DISTRICT',y='ANNUAL',width=0.5,edgecolor='k',align='center',linewidth=4,figsize=(16,8))
plt.xlabel('District',fontsize=30)
plt.ylabel('Annual Rainfall (in mm)',fontsize=20)
plt.title('Districts with Minumum Rainfall in India',fontsize=25)
ax.tick_params(labelsize=20)
plt.grid()
plt.ioff()


# Ladak which is part of Jammu and Kashmir receives 94.6 mm. Ladak and Kargil which receive less rainfall are part of Indian State Jammu and Kashmir.Jaisalmer,Sri Ganganaga and Barmer are part of Rajastan State.

# In[ ]:


Dist.groupby(['DISTRICT'])['ANNUAL'].max().sort_values().tail(5)


# In[ ]:


ax=Dist.groupby(['DISTRICT'])['ANNUAL'].max().sort_values().tail(20).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=4,figsize=(16,8))
#ax=k.plot.bar(x='DISTRICT',y='ANNUAL',width=0.5,edgecolor='k',align='center',linewidth=4,figsize=(16,8))
plt.xlabel('District',fontsize=30)
plt.ylabel('Annual Rainfall (in mm)',fontsize=20)
plt.title('Districts with Maximum Rainfall in India',fontsize=25)
ax.tick_params(labelsize=20)
plt.grid()
plt.ioff()


# Udupi which is part of Coastal Karnatka receives 4306 mm of Annual rainfall.Upper Siang is part of Arunachal Pradesh.East Kashi hill,Jaintha hills is part of Meghalaya.Tamenglong is part if Manipur.So Districts in Coastal Karnataka and North Eastern states receive heavy rainfall.

# In[ ]:


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
m=Basemap(projection='mill',llcrnrlat=0,urcrnrlat=40,llcrnrlon=50,urcrnrlon=100,resolution='c')
m.drawcoastlines()
m.drawcountries()
m.drawstates()
#m.fillcontinents()
m.fillcontinents(color='coral',lake_color='aqua')
#m.drawmapboundary()
m.drawmapboundary(fill_color='aqua')
#m.bluemarble()
#x, y = m(25.989836, 79.450035)
#plt.plot(x, y, 'go', markersize=5)
#plt.text(x, y, ' Trivandrum', fontsize=12);
lat,lon=13.340881,74.742142
x,y=m(lon,lat)
m.plot(x,y,'go')
plt.text(x, y, ' Udupi (4306mm)', fontsize=12);
lat,lon=28.879720,94.796970
x,y=m(lon,lat)
m.plot(x,y,'go')
plt.text(x, y, ' Upper Siang(4402mm)', fontsize=12);
"""lat,lon=25.578773,91.893257
x,y=m(lon,lat)
m.plot(x,y,'go')
plt.text(x, y, 'East Kashi Hills (6166mm)', fontsize=12);
lat,lon=25.389820,92.394913
x,y=m(lon,lat)
m.plot(x,y,'go')
plt.text(x, y, 'Jaintia Hills (6379mm)', fontsize=10);"""
lat,lon=24.987934,93.495293
x,y=m(lon,lat)
m.plot(x,y,'go')
plt.text(x, y, 'Tamenglong (7229mm)', fontsize=12);
lat,lon=34.136389,77.604139
x,y=m(lon,lat)
m.plot(x,y,'ro')
plt.text(x, y, ' Ladakh(94mm)', fontsize=12);
"""lat,lon=25.759859,71.382439
x,y=m(lon,lat)
m.plot(x,y,'ro')
plt.text(x, y, ' Barmer(268mm)', fontsize=12);"""
lat,lon=26.915749,70.908340
x,y=m(lon,lat)
m.plot(x,y,'ro')
plt.text(x, y, ' Jaisalmer(181mm)', fontsize=12);
plt.title('Places with Heavy and Scanty Rainfall in India',fontsize=20)
plt.ioff()
plt.show()


# Green Dots are areas of heavy rainfall and Red Dots are places of scanty rainfall.

# **Which Years Had Highest Rainfall in India ?**

# In[ ]:


India.groupby("YEAR").mean()['ANNUAL'].sort_values(ascending=False).head(10)


# **Which Years Had Low Rainfall in India?**

# In[ ]:


India.groupby("YEAR").mean()['ANNUAL'].sort_values(ascending=False).tail(10)

