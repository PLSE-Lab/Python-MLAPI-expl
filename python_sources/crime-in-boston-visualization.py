#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/crime.csv",  encoding = "ISO-8859-1") # read the excel file


# In[ ]:


df.head()


# In[ ]:


# Convert OCCURED_ON_DATE to datetime
df['OCCURRED_ON_DATE'] = pd.to_datetime(df['OCCURRED_ON_DATE'])
# Remove unused columns
df = df.drop(['INCIDENT_NUMBER','STREET','Location'], axis=1)

# Fill in nans in SHOOTING column
df.SHOOTING.fillna('N', inplace=True)

# Replace -1 values in Lat/Long with Nan
df.Lat.replace(-1, None, inplace=True)
df.Long.replace(-1, None, inplace=True)


# In[ ]:


#df.columns.values
df.shape  # No. of rows and columns


# In[ ]:


sns.countplot("SHOOTING", hue="DISTRICT", data = df) # Which Year has reported more number of shootings


# In[ ]:


df.boxplot('OFFENSE_CODE','DISTRICT',rot=30,figsize=(6,6)) # District wise offence occurances


# In[ ]:


import matplotlib.pyplot as plt
sns.countplot("DISTRICT", hue="SHOOTING", data = df) # Which district has reported more number of shootings
plt.xlabel('DISTRICT', fontsize=12)
plt.title('Frequency shootings in district wise')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.show()


# In[ ]:


df_count= df['HOUR'].value_counts()
sns.set(style="darkgrid")
sns.barplot(df_count.index,df_count.values,alpha=1) # Find which hour the crime is more
plt.xlabel('HOUR', fontsize=12)
plt.title('Frequency Crime in Hour wise')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.show()


# In[ ]:


# Percentage in year wise
labels = df['YEAR'].astype('category').cat.categories.tolist()
counts = df['YEAR'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels,  autopct='%1.1f%%',shadow=True) 
ax1.axis('equal')
plt.show()


# In[ ]:



#data = df.drop(['INCIDENT_NUMBER'], axis=1) #dropping 'Id' column from DataFrame
newDf = df.filter(['YEAR','MONTH','HOUR','UCR_PART'], axis=1)
newDf.head()


# In[ ]:


sns.pairplot(newDf,hue ='UCR_PART') # UCR wise report


# In[ ]:


df_Warrent_Arrent = df[df.OFFENSE_CODE_GROUP=='Warrant Arrests'] 
sns.kdeplot(df_Warrent_Arrent.YEAR) # Year wise warrant arrests


# In[ ]:


sns.distplot(df['HOUR'])


# In[ ]:


sns.jointplot(x='MONTH',y='YEAR',data=df,kind='hex', gridsize=20)


# In[ ]:


sns.kdeplot(df['MONTH'], df['YEAR'] )


# In[ ]:


sns.violinplot(x='UCR_PART',y='YEAR',data=df)


# In[ ]:


# Keep only data from complete years (2016, 2017)
data = df.loc[df['YEAR'].isin([2016,2017])]
sns.scatterplot(x='Lat',
               y='Long',
                alpha=0.01,
               data=data)


# In[ ]:


# Keep only data from complete years (2016, 2017) 
sns.scatterplot(x='Lat',
               y='Long',
                hue='DISTRICT',
                alpha=0.01,
               data=data)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)


# In[ ]:


import folium
from folium.plugins import HeatMap
# Create basic Folium crime map
crime_map = folium.Map(location=[42.3125,-71.0875], 
                       tiles = "Stamen Toner",
                      zoom_start = 11)

# Add data for heatmp 
data_heatmap = data[data.YEAR == 2017]
data_heatmap = data[['Lat','Long']]
data_heatmap = data.dropna(axis=0, subset=['Lat','Long'])
data_heatmap = [[row['Lat'],row['Long']] for index, row in data_heatmap.iterrows()]
HeatMap(data_heatmap, radius=10).add_to(crime_map)

# Plot!
crime_map


# In[ ]:


# District wise count
sns.catplot(y='DISTRICT',
           kind='count',
            height=8, 
            aspect=1.5,
            order=data.DISTRICT.value_counts().index,
           data=data)


# In[ ]:



#Day wise crime
import pylab
pylab.rcParams['figure.figsize'] = (14.0, 8.0)    

data = df.loc[df['YEAR'].isin([2016,2017])]    

Sunday = data[data['DAY_OF_WEEK'] == "Sunday"]
Monday = data[data['DAY_OF_WEEK'] == "Monday"]
Tuesday = data[data['DAY_OF_WEEK'] == "Tuesday"]
Wednesday = data[data['DAY_OF_WEEK'] == "Wednesday"]
Thursday = data[data['DAY_OF_WEEK'] == "Thursday"]
Friday = data[data['DAY_OF_WEEK'] == "Friday"]
Saturday = data[data['DAY_OF_WEEK'] == "Saturday"]

ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
ax1.plot(data.groupby('DAY_OF_WEEK').size(), 'ro-')
ax1.set_title ('All crimes - 2016 & 2017')
start, end = ax1.get_xlim()
ax1.xaxis.set_ticks(np.arange(start, end, 1))
    
ax2 = plt.subplot2grid((3,3), (1, 0))
ax2.plot(Sunday.groupby('HOUR').size(), 'o-')
ax2.set_title ('Sunday')

ax2 = plt.subplot2grid((3,3), (1, 1))
ax2.plot(Monday.groupby('HOUR').size(), 'o-')
ax2.set_title ('Monday')

ax2 = plt.subplot2grid((3,3), (1, 2))
ax2.plot(Tuesday.groupby('HOUR').size(), 'o-')
ax2.set_title ('Tuesday')

ax2 = plt.subplot2grid((3,3), (2, 0))
ax2.plot(Wednesday.groupby('HOUR').size(), 'o-')
ax2.set_title ('Wednesday')

ax2 = plt.subplot2grid((3,3), (2, 1))
ax2.plot(Thursday.groupby('HOUR').size(), 'o-')
ax2.set_title ('Thursday')

ax2 = plt.subplot2grid((3,3), (2, 2))
ax2.plot(Friday.groupby('HOUR').size(), 'o-')
ax2.set_title ('Friday')

plt.tight_layout(2)
plt.show()


# In[ ]:


# Bar plot
y = np.empty([6,7])
h = [None]*7
width = 0.1
daysOfWeekIdx = data.groupby('DAY_OF_WEEK').size().keys()
daysOfWeekLit = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
color_sequence = ['#1f77b4', '#ff7f0e', '#2ca02c','#d62728', '#9467bd', '#8c564b']
for j in range(0,6):
    y[j] =data[data['DAY_OF_WEEK'] == daysOfWeekLit[j]].groupby('DAY_OF_WEEK').size().get_values()

for i in range(0,6):
   h[i] = ax2.bar(5, y[i], width, color=color_sequence[i], alpha = 0.7)
plt.show()


# In[ ]:


daysOfWeekIdx


# In[ ]:




