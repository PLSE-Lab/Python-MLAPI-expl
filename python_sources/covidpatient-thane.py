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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime
import folium
from sklearn.cluster import KMeans
import matplotlib as mpl


# Description:
# This data provides information about the COVID-19 about the patients, containment areas, Hospital details, Quarantine facilities and Food shelter distribution details etc.
# 
# Released Under:
# National Data Sharing and Accessibility Policy (NDSAP)
# 
# Contributor:
# Maharashtra Thane
# 
# Sectors:
# HealthFamily WelfareHealth and Family welfare
# 
# Published on Data Portal:
# June 30, 2020

# In[ ]:


data = pd.read_csv('/kaggle/input/COVIDPatient.csv')


# In[ ]:


data.head()


# * lets remove the unneccessary colums which are 'State', 'City Name' and 'Contact Determined' contains only one value each. i.e. Maharastra Thane and Under Investigation Respectively
# * we are removing 'Locally Acquired Yes/No', 'Acquired Overseas: Yes/No' columns as well because all patients acquired corona Locally

# In[ ]:


unneccesary_cols = ['State', 'City Name','Locally Acquired Yes/No', 'Acquired Overseas: Yes/No', 'Contact Determined']


# In[ ]:


data_ = data.drop(unneccesary_cols, axis=1)


# In[ ]:


data_.info()


# # Age Columns dtype is Object lets convert it to float 

# 1. first converted "Not Available" values in Age column to "0"
# 2. converted dtype object to float
# 3. replaced all zero values with mean of Age.

# In[ ]:


age = data_["Age"]
for i in range(0,7054):
    if age[i] == 'Not Available' :
        age[i] = '0'
        
data_["Age"] = data_["Age"].astype(str).astype(float)

age = data_["Age"]
for i in range(0,7054):
    if age[i] == '0' :
        age[i] = age.mean()


# In[ ]:


data_.info()


# # Lets check the number of positive cases with respect to Gender

# In[ ]:


sns.countplot(data['Gender'],hue = data['Current Status'])
data['Gender'].value_counts()


# # Lets check the current status of the patients

# In[ ]:


sns.countplot(data['Current Status'])
data['Current Status'].value_counts()


# # lets see number of patients in various containment area wards

# In[ ]:


plt.figure(figsize=(20,10))

sns.countplot(data['Ward Name'])
plt.xticks(rotation=45)
data['Ward Name'].value_counts()


# In[ ]:


sns.catplot(x="Current Status", y="Age", kind="violin", data=data_)


# we can see that as age increases discharge rate decreases and death rate increases 

# * we have Result Date column values in 2 formats containg - and / in between 
# * lets replace all - in Result Date to /
# * lets sort data_ with respect to Result Dates

# In[ ]:


import re
data_["Result Date"] = data_["Result Date"].replace(to_replace ='-', value = '/', regex = True) 


# In[ ]:


data_['Result Date'] = pd.to_datetime(data_['Result Date'],errors='coerce').dt.date


# In[ ]:


df = data_.sort_values('Result Date')


# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(df['Result Date'])
plt.xticks(rotation=90)


# # Let's use the longitude and latitude values so that we can plot them and get insights on clustering.

# In[ ]:


plt.figure(figsize=(15,10))
plt.plot(data['Latitude'], data['Longitude'],'.')
plt.title('Plot of patient')
plt.show()


# # Testing Different Numbers of Clusters
# We will try out different number of clusters using KMeans clustering and then decide what the best number of clusters is by seeing when the SSE score stops improving significantly.

# In[ ]:


def get_kmeans_score(data_, center):
    '''
    returns the kmeans score regarding SSE for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the SSE score for the kmeans model fit to the data
    '''
  
    kmeans = KMeans(n_clusters=center)
    model = kmeans.fit(data_)
    score = np.abs(model.score(data_))
    
    return score

scores = []
centers = list(range(1,25))

for center in centers:
    scores.append(get_kmeans_score(data.loc[: ,['Longitude','Latitude']], center))
    
    
plt.plot(centers, scores, linestyle='--', marker='o', color='b');
plt.xlabel('Number of Centers');
plt.ylabel('SSE');
plt.title('SSE vs. Number');


# So it looks like we have 4 clusters, according to the elbow method, since after 4 there is insignificant improvement. Now let's find their coordinates so that we can plot them on the map as well.

# In[ ]:


kmeans = KMeans(n_clusters=4)

kmeans.fit(data.loc[: ,['Longitude','Latitude']]).cluster_centers_


# In[ ]:


m = folium.Map(location=[19.198741,72.977948])


# In[ ]:


folium.Marker([19.19784481, 72.98635878],popup = 'Cluster Center', icon=folium.Icon(color='red') ).add_to(m)
folium.Marker([19.17479723, 73.03017528],popup = 'Cluster Center', icon=folium.Icon(color='red') ).add_to(m)
folium.Marker([19.19837351, 72.95482727],popup = 'Cluster Center', icon=folium.Icon(color='red') ).add_to(m)
folium.Marker([19.2414287 , 72.97456634],popup = 'Cluster Center', icon=folium.Icon(color='red') ).add_to(m)


# In[ ]:


m

