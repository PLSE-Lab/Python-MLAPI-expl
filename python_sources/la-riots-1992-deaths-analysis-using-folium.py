#!/usr/bin/env python
# coding: utf-8

# # Background
# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSq1nErz3DSY70KK_pIBgBVeNrXbn7NXr5tfomCAAF0f0VotIbt&usqp=CAU)
# 
# The 1992 LA Riots occurred in Los Angeles County during April and May of 1992.  There were already underlying ethnic tensions, but the riots were sparked by the jury acquittal of the 4 officers responsible for usage of excessive force in the beating and arrest of Rodney King.  Ultimately the riots resulted in 63 deaths, thousands of injuries, and $1 billion in damages.  In this dataset, we have information on the 63 deaths that occurred in relation the LA Riots.  We will be plotting them using Folium to see if there are any patterns with their locations. 
# 

# Importing packages

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import folium
from sklearn.cluster import KMeans

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Basic Information on the Dataset

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

#importing the data
filename = '../input/los-angeles-1992-riot-deaths-from-la-times/la-riots-deaths.csv'
data = pd.read_csv(filename)
#Exploring the data
data.info()
data.head()


# As we can see the dataset gives us basic information on each of the deaths, such as their name, age, race, location, etc...Let's extract the longitude and latitude values so that we can plot them and get insights on clustering.

# In[ ]:



#Here I am switching latitude and longitude because it is incorrectly labeled in the dataset.
longitude = data['lat'].values
latitude = data['lon'].values


# In[ ]:



m = folium.Map(location=[34.0593, -118.274], zoom_start = 10)

# I skipped 51 because that one showed up as a NaN.
for i in range(63):
    if (i != 51):
        folium.Marker([latitude[i],longitude[i]],popup = data.loc[i,'Full Name'] ).add_to(m)
m        


# In[ ]:


plt.plot(data['lat'], data['lon'],'.')
plt.title('Plot of Deaths')
plt.show()


# ### Testing Different Numbers of Clusters
# We will try out different number of clusters using KMeans clustering and then decide what the best number of clusters is by seeing when the SSE score stops improving significantly.

# In[ ]:


#I decided to just fill in the missing value for the following kmeans clustering.
values = {'lat': 34.0593, 'lon':-118.274}
filleddata = data.fillna(value=values)


# In[ ]:


def get_kmeans_score(data, center):
    '''
    returns the kmeans score regarding SSE for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the SSE score for the kmeans model fit to the data
    '''
  
    kmeans = KMeans(n_clusters=center)
    model = kmeans.fit(data)
    score = np.abs(model.score(data))
    
    return score

scores = []
centers = list(range(1,11))

for center in centers:
    scores.append(get_kmeans_score(filleddata.loc[: ,['lon','lat']], center))
    
plt.plot(centers, scores, linestyle='--', marker='o', color='b');
plt.xlabel('Number of Centers');
plt.ylabel('SSE');
plt.title('SSE vs. Number');


# So it looks like we have two clusters, according to the elbow method, since after 2 there is insignificant improvement. Now let's find their coordinates so that we can plot them on the map as well.

# In[ ]:


kmeans = KMeans(n_clusters=2)

kmeans.fit(filleddata.loc[: ,['lon','lat']]).cluster_centers_
#I am not sure why it's switching the coordinates for the first cluster


# In[ ]:



#Plotting the two centers 
folium.Marker([34.02756075,-118.28079299],popup = 'Cluster Center', icon=folium.Icon(color='red') ).add_to(m)
folium.Marker([34.0593,-118.274],popup = 'Cluster Center', icon=folium.Icon(color='red') ).add_to(m)

m


# The two main clusters that the KMeans algorithm found were one centered around University Park and one centered around Westlake. 

# ## Further Studies and Finishing Thoughts
# This was my very first notebook and I was just excited to publish it, so I welcome all comments and suggestions!  To further study this data, I could try out different clustering algorithms besides KMeans.  Also, I could look at a Racial breakdown of the deaths. 
