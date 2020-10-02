#!/usr/bin/env python
# coding: utf-8

# # Loading data to MongoDB
# Since the dataset is rather large, I thought some people might want something that could load the data to a database rather quickly.
# If you have MongoDB installed on your computer and running, you can run each of the cells (except the last) in this notebook to import all the data to your local database.

# In[ ]:


import hdbscan
from matplotlib import pylab
import pymongo as pm
import numpy as np
import pandas as pd
import json
import csv
from geopy import distance
from descartes import PolygonPatch


# ### Connect to your MongoDB instance and access the `nyc` databse and `taxi_data` collection.

# In[ ]:


connection = pm.MongoClient()
nyc = connection.nyc
taxi_collection = nyc.taxi_data


# ### Create indices for the `taxi_collection` called 'pickup_location' and 'dropoff_location'.

# In[ ]:


taxi_collection.drop()
taxi_collection.create_index([("pickup_location",  pm.GEO2D)])
taxi_collection.create_index([("dropoff_location",  pm.GEO2D)])


# ### Read and iterate over the file.
# This function also combine the longitude and latitudes and sets them to the location indices we made in the last cell.

# In[ ]:


def import_content(filename):
    print('Reading file...')
    data = open(filename, 'r')
    print('File read...')
    header = True
    keys = []
    step = 0
    batch_data = []
    for line in data:
        line_data = {}
        if header:
            keys = line.split(',')
            print(keys)
            header = False
        else:
            line = line.split(',')
            for k in range(len(keys)):
                line_data[keys[k]] = line[k]
            try:
                line_data['pickup_location'] = line_data['pickup_longitude'] + ',' + line_data['pickup_latitude'] 
                line_data['dropoff_location'] = line_data['dropoff_longitude'] + ',' + line_data['dropoff_latitude']
            except:
                pass # sometimes there are is no data in the pickup or dropoff locs
            if step % 100000 == 0:
                batch_data = load_many_and_report(taxi_collection, batch_data, step)
            else:
                batch_data += [line_data]
        step += 1
    batch_data = load_many_and_report(taxi_collection, batch_data, step)
    print('Finished loading data.')


# #### Common loader function.

# In[ ]:


def load_many_and_report(coll, batch_data, step):
    coll.insert_many(batch_data)
    print('%d lines loaded...' % step)
    batch_data = []
    return batch_data


# ## Now just call the function!

# In[ ]:


# import_content('data/train.csv')
import_content('../input/train.csv')


# # Extra work...
# Here is some extra work I did. It's not very impressive, but it shows you how you can pull data back out of the database to continue your work.

# #### Function to calcuate the L2 distance (straight-line).

# In[ ]:


'''
Knowing that you're receiving location points as (long, lat) and you need to swap them.
'''
def getLength(loc1, loc2):
    return distance.distance((loc1[1], loc1[0]), (loc2[1], loc2[0])).miles


# ### Querying data out of the database.
# Here I just use the basic `find()` function, and this will just give you a `cursor` (`iterator`). For help for better queries, you can go [**here**](https://docs.mongodb.com/master/tutorial/getting-started/#query-documents).
# 
# #### Note: This could have been performed eariler in the notebook, but it is done here for demo purposes.

# In[ ]:


cursor = taxi_collection.find()
i = 1
while cursor.alive:
    i += 1
    if i % 100000 == 0:
        print('%d distances calcuated...' % i)
    taxi = cursor.next()
    D = getLength(taxi['pickup_location'].split(','), taxi['dropoff_location'].split(','))
    taxi_collection.update_one( {"_id":taxi['_id']}, {"$set": { "l2_distance": D }} )

