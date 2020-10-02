#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# for data visualizations
import matplotlib.pyplot as plt
import seaborn as sns

import chardet

# Any results you write to the current directory are saved as output.


# In[ ]:


# look at the first ten thousand bytes to guess the character encoding

with open("../input/PakistanSuicideAttacks Ver 11 (30-November-2017).csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

# check what the character encoding might be
print(result)


# In[ ]:


data1 = pd.read_csv('../input/PakistanSuicideAttacks Ver 11 (30-November-2017).csv', encoding='Windows-1252')
data2 = pd.read_csv('../input/PakistanSuicideAttacks Ver 6 (10-October-2017).csv', encoding='Windows-1252')

print(data1.shape)
print(data2.shape)


# In[ ]:


# checking the columns in the data1

data1.columns


# In[ ]:


# checking the columns in the data2

data2.columns


# In[ ]:


# as both the datasets contain similar columns we can concatenate them

data = pd.concat([data1, data2])

# checking the shape of new dataset
data.shape


# In[ ]:


data1.sample(10)


# In[ ]:


data.describe()


# In[ ]:


# checking if it contains aby NULL values

data.isnull().any()


# In[ ]:


# checking the sector target type

data['Targeted Sect if any'].value_counts()

# replacing shiite with Shiite and Shiite/sunni with Shiite
data['Targeted Sect if any'].replace(('shiite', 'Shiite/sunni'), ('Shiite', 'Shiite'), inplace  = True)

size = [697, 96, 76, 18, 2, 2]
colors = ['orange', 'pink', 'crimson', 'purple', 'violet', 'red']
labels= ['None', 'Shiite', 'Sunni', 'Christian', 'Ahmedi', 'Jews']
explode = [0, 0, 0, 0, 1, 2]

my_circle = plt.Circle((0, 0), 0.7, color = 'white')

plt.pie(size, colors = colors, labels = labels,explode = explode, shadow = False, autopct = '%.2f%%')
plt.title('Targeted Sectors by the Terrorists', fontsize = 20)
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.show()


# In[ ]:


# checking the target type

data['Target Type'].value_counts().plot.bar(figsize = (15, 7), color = 'darkblue')
plt.title('Targets set by the Terrorists', fontsize = 20)
plt.xlabel('Targets')
plt.ylabel('Count')
plt.show()


# In[ ]:


# checking the influencing event

data['Influencing Event/Event'].value_counts().head(10)


# In[ ]:


# checking the open/closed space

data['Open/Closed Space'].value_counts()

# replacing open and closed with Open and Closed
data['Open/Closed Space'].replace(('open', 'closed', 'Open/Closed', 'Open '),('Open', 'Closed', 'Open', 'Open'), inplace = True)

data['Open/Closed Space'].value_counts().plot.pie(colors = ['lightgreen', 'orange'], explode = [0, 0.1])
plt.title('A pie Chart representing choice of terrorists for bombing', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()


# In[ ]:


# checking the Location Sensitivity

data['Location Sensitivity'].value_counts()

# replacing low with Low
#data['Location Sensitivity'].replace('low', 'Low', inplace = True)

# plotting a pie chart

size = [528, 239, 149]
colors = ['violet', 'orange', 'pink', 'lightgreen']
labels = ['High', 'Low', 'Medium']
explode = [0, 0, 0.1]

plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%')
plt.legend()
plt.show()


# In[ ]:


# checking the different locations category

data['Location Category'].value_counts(dropna = False).head(15).plot.bar(figsize = (15, 7))
plt.title('Top 15 Locations for Suicide Bombing', fontsize = 20)
plt.xlabel('Locations')
plt.ylabel('No. of Suicide Bombings')
plt.show()


# In[ ]:


# checking the Locations

data['Location'].value_counts().head(10)


# In[ ]:


# checkig the longitudes

data['Longitude'].value_counts().head()


# In[ ]:


# chekcing the latitudes

data['Latitude'].value_counts().head()


# In[ ]:


# checking the time at which the blast occured

data['Time'].value_counts().head(20)


# In[ ]:


# checking the types of holidays while on suicide bombings

data['Holiday Type'].value_counts(dropna = False)


# In[ ]:


# checking the various typs of Islamic dates

data['Islamic Date'].value_counts().head(15)


# In[ ]:


# checking the various blast day types

data['Blast Day Type'].value_counts()

# making a pie chart of probability of bombings on type of holidays

size = [801, 156, 10]
labels = ['Working Day', 'Holiday', 'Weekend']
colors = ['lightblue', 'violet', 'magenta']
explode = [0, 0, 0.1]

plt.pie(size, colors = colors, labels = labels, shadow = True, explode = explode, autopct = '%.2f%%')
plt.legend()
plt.show()


# In[ ]:


# filling missing values

data['Islamic Date'].fillna('None', inplace = True)
data['Blast Day Type'].fillna(data['Blast Day Type'].mode()[0], inplace = True)
data['Holiday Type'].fillna('None', inplace = True)
data['Time'].fillna(data['Time'].mode()[0], inplace = True) # as it a common value
data['Latitude'].fillna(34.0043, inplace = True) # as it is a common value
data['Longitude'].fillna(71.5448, inplace = True) # as it is most common
data['Location'].fillna(data['Location'].mode()[0], inplace = True)
data['Location Category'].fillna(data['Location Category'].mode()[0], inplace = True)
data['Location Sensitivity'].fillna(data['Location Sensitivity'].mode()[0], inplace = True)
data['Open/Closed Space'].fillna(data['Open/Closed Space'].mode()[0], inplace = True)
data['Influencing Event/Event'].fillna('None', inplace = True)
data['Target Type'].fillna('None', inplace = True)
data['Targeted Sect if any'].fillna('None', inplace = True)
data['Killed Min'].fillna(0, inplace = True)
data['Killed Max'].fillna(1.0, inplace = True)
data['Injured Min'].fillna(data['Injured Min'].mode()[0], inplace = True)
data['Injured Max'].fillna(data['Injured Min'].mode()[0], inplace = True)
data['No. of Suicide Blasts'].fillna(1.0, inplace = True)
data['Explosive Weight (max)'].fillna(data['Explosive Weight (max)'].mode()[0], inplace = True)
data['Hospital Names'].fillna(data['Hospital Names'].mode()[0], inplace = True)
data['Temperature(C)'].fillna(data['Temperature(C)'].mode()[0], inplace = True)
data['Temperature(F)'].fillna(data['Temperature(F)'].mode()[0], inplace = True)


# In[ ]:


# visualizing minimum no. of people injured in a bomb blast

sns.distplot(data['Injured Min'], color = 'pink')
plt.title('Minimum no. of people injured in a blast', fontsize = 20)
plt.xlabel('Blasts in different days')
plt.ylabel('Count')
plt.show()


# In[ ]:


# checking top 15 preferred explosive weights

data['Explosive Weight (max)'].value_counts().head(15).plot.bar()
plt.title('Most Preferred Explosive Weights in general', fontsize = 15)
plt.xlabel('Explosive Weights')
plt.ylabel('Count')
plt.show()


# In[ ]:


# checking the Provinces where suicide bombing happened

data['Province'].value_counts().plot.bar(figsize = (20, 7), color = 'lightblue')
plt.title('Provinces with hhighest number of suicide bombing instances')
plt.xlabel('Province')
plt.ylabel('count')
plt.show()


# In[ ]:


# locations where suicide bombings took place most often

data['Location'].value_counts().head(15).plot.bar(figsize = (20, 7), color = 'violet')
plt.title('Top 15 Locations suffering from highest Suicide Bombings', fontsize = 20)
plt.xlabel('Different Locations')
plt.ylabel('No. of Suicide Attacks')
plt.show()


# In[ ]:


# checking the no. of blasts occuring in a single day

data1['No. of Suicide Blasts'].value_counts(dropna = False)

labels = ['1', '2', '3', '4']
colors = ['blue', 'lightgreen', 'lightpink', 'magenta']
size = [375, 32, 5, 2]
explode = [0.1, 0.1, 0.2, 0.3]

plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True)
plt.axis('off')
plt.title('Suicide Blasts Per Day', fontsize = 20)
plt.legend()
plt.show()


# In[ ]:


# visualizing minimum killed in a bombing attack

sns.distplot(data['Killed Min'], color = 'yellow')
plt.title('Minimum no. of People Killed in a Suicide Bombing Attack', fontsize = 20)
plt.xlabel('Different Suicide Attacks')
plt.ylabel('count')
plt.show()


# In[ ]:


# visualizing maximum killed in a bombing attack

sns.distplot(data['Killed Max'], color = 'orange')
plt.title('Maximum no. of People Killed in a Suicide Bombing Attack', fontsize = 20)
plt.xlabel('Different Suicide Attacks')
plt.ylabel('count')
plt.show()


# In[ ]:


# Top 10 Hospital Names

data1['Hospital Names'].value_counts().head(10)


# In[ ]:


# total no. of different cities where suicide bombing has Occured

x = data['City'].value_counts().nunique()
print("Total no. of different cities where suicide bombing has Occured: ", x)


# In[ ]:


# Top 20 Most Preferred Cities for Suicide Bombing

data1['City'].value_counts().head(15).plot.bar(figsize = (18, 7), color = 'purple')
plt.title('Top 15 Most Preferred Cities for Suicide Bombing', fontsize = 20)
plt.xlabel('Names of Cities')
plt.ylabel('Count')
plt.show()


# In[ ]:




