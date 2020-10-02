#!/usr/bin/env python
# coding: utf-8

# I believe we must do a better job wrt the latitude and longitude, better than "distance from centre" ! Reverse geocoding gives us the administrative name of the region of the rental property. 
# 
# Note: I've used the open-source package "[reverse_geocoder][1]", which was way faster than geopy. Kaggle doesn't support it though, commenting it for correctness' sake -- run it on your system to see the results. 
# 
#   [1]: https://github.com/thampiman/reverse-geocoder

# In[ ]:


import pandas as pd
import csv
#import reverse_geocoder as rg

train_file =  "../input/train.json"


# Getting the data frame ready for processing, i'll be doing it only for the training set. 

# In[ ]:


train_df = pd.read_json(train_file)

train_coords = train_df[["listing_id", "latitude", "longitude"]]


# Reverse Geocoder takes a list of tuples(of latitude and longitude) as its input.

# In[ ]:


lat_lon = []
listings = []

for i, j in train_coords.iterrows():
    lat_lon.append((j["latitude"], j["longitude"]))
    listings.append(int(j["listing_id"]))


# rg.search([Lat1, Lon1), (Lat2, Lon2), ....])

# In[ ]:


#results = rg.search(lat_lon) #Uncomment this. This is the juice!
results = [] #Comment this :(

nbd = [[listings[i], results[i]['name']] for i in range(0, len(results))] #getting ready to write to csv 


# In[ ]:


with open("neighborhood_train.csv", "wb") as f:

    writer = csv.writer(f, delimiter = ",")
    writer.writerows(nbd)


# Hope the neighborhood variable is of use to you! 
