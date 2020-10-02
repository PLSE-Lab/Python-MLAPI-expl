#!/usr/bin/env python
# coding: utf-8

# # Us mass shootings analysis
# ** Cleuton Sampaio **
# This is the first attempt to analyze this dataset, and I'll do it trying to save lifes, nothing else. This dataset is published on Kaggle: https://www.kaggle.com/zusmani/us-mass-shootings-last-50-years
# 

# ## 1. Getting and viewing the Dataset
# This kernel uses a different dataset I created, which has the NaN coordinates problem fixed. See in the "Data" Section.

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

rawdata = pd.read_csv('../input/us-mass-shootings-nan-coordinates-fixed/mass_shootings_dataset_coords_fixed.csv',encoding = 'ISO-8859-1', parse_dates=['Date'])


# In[ ]:


rawdata.head()


# ## 2. Total deaths per year
# 

# In[ ]:


rawdata[['Date', 'Total victims']].groupby([(rawdata.Date.dt.year)])['Total victims'].sum()


# OMG! Look at 2017! This is the impact of October 2017 Las Vegas shooting... Very sad.

# In[ ]:


rawdata[(rawdata.Date.dt.year==2017)]


# ## 3. Deaths vs U.S. Map
# To plot deaths we could just use the coordinates...
# 

# In[ ]:


coords = rawdata[['Longitude', 'Latitude']].dropna()
coords.plot(kind='scatter',x='Longitude',y='Latitude')


# Looks like the US Map... Let's improve it. Wait... That lonelly point in the upper left corner is in Alaska?
# 

# In[ ]:


rawdata[(rawdata.Latitude > 60)]


# Yup! So we need a map of US that includes Alaska... It will not fit exactly but...

# Hmmmm... And that other lonelly point in the lower left corner? Is that Hawaii?
# 

# In[ ]:


rawdata[(rawdata.Latitude < 25)]


# In[ ]:


rawdata[(rawdata.Longitude > -85)].max()


# In[ ]:



from scipy.misc import imread
import matplotlib.cbook as cbook
rawdata.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.4,
    s=rawdata["Total victims"]*1.5, label="Victims", figsize=(20,9),
    c="Total victims", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()


# Matplotlib has a Map feature (Basemap) but is not working. I've tried to adjust the scale to insert a Map background, but the effect is too bad. If you look at the points you can see the map of US and the total people affected is the size of the mark.
# I have used the following code to access Google Maps API in order to get right coordinates from the ** Location ** attribute:

# In[ ]:


'''
import urllib.request
import json
url = 'https://maps.googleapis.com/maps/api/geocode/json?address=Kalamazoo&key=SORRY-USE-YOURS'
req = urllib.request.Request(url)

r = urllib.request.urlopen(req).read()
response = json.loads(r.decode('utf-8'))

for item in response['results']:
    print("Latitude:", item['geometry']['location']['lat'], 
          "Longitude:",item['geometry']['location']['lng'] )
'''


# In[ ]:


missing=rawdata[(np.isnan(rawdata.Latitude)) | (np.isnan(rawdata.Longitude))]


# I have ran the code bellow to get the right coordinates:

# In[ ]:


'''
import urllib.parse
def get_location(address):
    #print('input address:',address)
    encoded=urllib.parse.quote_plus(address)
    print('Encoded:',encoded,'address:',address)
    url = 'https://maps.googleapis.com/maps/api/geocode/json?address=' + encoded +'&key=[USE YOUR API KEY]'
    req = urllib.request.Request(url)
    r = urllib.request.urlopen(req).read()
    response = json.loads(r.decode('utf-8'))
    results = response['results']
    print('latitude:',results[0]['geometry']['location']['lat'], 'longitude:',results[0]['geometry']['location']['lng'])
    if len(results) > 0:
        return results[0]['geometry']['location']['lat'], results[0]['geometry']['location']['lng']
    else:
        return 0,0

coords=[]
for address in missing['Location']:
    lat,long = get_location(address)
    coords.append((lat,long))
coords
'''


# In[ ]:


#tuplas_coord = pd.DataFrame(coords, columns=['Latitude', 'Longitude'])


# In[ ]:


#rawdata.loc[(np.isnan(rawdata.Latitude)) | (np.isnan(rawdata.Longitude)), ['Latitude', 'Longitude']] = tuplas_coord[['Latitude', 'Longitude']]


# In[ ]:


#rawdata


# In[ ]:


#rawdata = pd.read_csv('mass_shootings_dataset_coords_fixed.csv',encoding = 'ISO-8859-1', parse_dates=['Date'])


# In[ ]:





# Now let's plot a better map...

# In[ ]:


rawdata.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.4,
    s=rawdata["Total victims"]*1.5, label="Victims", figsize=(20,9),
    c="Total victims", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()


# Looking at the map, we can see that the murders are scattered all over the country, but there seem to be more cases in the eastern US, but there are cases with a large number of victims in the West. In the East they are more scattered and in the West they are more concentrated.
# I'll do a numerical analysis later ...

# The author asks for some correlation studies between Shooter and his/hers race or gender. Well, this is a very controversial approach, which can lead to hasty conclusions. Correlations do not imply Causality.
# But we can do a study based on the geographic data, which are very significant. Why is there such a difference between East and West cases? Would it have anything to do with the study pointing to Seth Stephens-Davidowitz in his book "Everybody Lies"?
