#!/usr/bin/env python
# coding: utf-8

# <p style="font-family: Arial; font-size:2.75em;color:purple; font-style:bold"><br>
# 
# Clustering with scikit-learn
# 
# <br><br></p>

# In this notebook, we will learn how to perform k-means lustering using scikit-learn in Python. 
# 
# We will use cluster analysis to generate a big picture model of the weather at a local station using a minute-graunlarity data. In this dataset, we have in the order of millions records. How do we create 12 clusters our of them?
# 

# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold"><br>
# 
# Importing the Necessary Libraries<br></p>

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from itertools import cycle, islice
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

get_ipython().run_line_magic('matplotlib', 'inline')


# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold"><br>
# 
# Creating a Pandas DataFrame from a CSV file<br><br></p>
# 

# In[ ]:


data = pd.read_csv('../input/minute_weather.csv')


# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold">Minute Weather Data Description</p>
# <br>
# The **minute weather dataset** comes from the same source as the daily weather dataset that we used in the decision tree based classifier notebook. The main difference between these two datasets is that the minute weather dataset contains raw sensor measurements captured at one-minute intervals. Daily weather dataset instead contained processed and well curated data. The data is in the file **minute_weather.csv**, which is a comma-separated file.
# 
# As with the daily weather data, this data comes from a weather station located in San Diego, California. The weather station is equipped with sensors that capture weather-related measurements such as air temperature, air pressure, and relative humidity. Data was collected for a period of three years, from September 2011 to September 2014, to ensure that sufficient data for different seasons and weather conditions is captured.
# 
# Each row in **minute_weather.csv** contains weather data captured for a one-minute interval. Each row, or sample, consists of the following variables:
# 
# * **rowID:** 	unique number for each row	(*Unit: NA*)
# * **hpwren_timestamp:**	timestamp of measure	(*Unit: year-month-day hour:minute:second*)
# * **air_pressure:** air pressure measured at the timestamp	(*Unit: hectopascals*)
# * **air_temp:**	air temperature measure at the timestamp	(*Unit: degrees Fahrenheit*)
# * **avg_wind_direction:**	wind direction averaged over the minute before the timestamp	(*Unit: degrees, with 0 means coming from the North, and increasing clockwise*)
# * **avg_wind_speed:**	wind speed averaged over the minute before the timestamp	(*Unit: meters per second*)
# * **max_wind_direction:**	highest wind direction in the minute before the timestamp	(*Unit: degrees, with 0 being North and increasing clockwise*)
# * **max_wind_speed:**	highest wind speed in the minute before the timestamp	(*Unit: meters per second*)
# * **min_wind_direction:**	smallest wind direction in the minute before the timestamp	(*Unit: degrees, with 0 being North and inceasing clockwise*)
# * **min_wind_speed:**	smallest wind speed in the minute before the timestamp	(*Unit: meters per second*)
# * **rain_accumulation:**	amount of accumulated rain measured at the timestamp	(*Unit: millimeters*)
# * **rain_duration:**	length of time rain has fallen as measured at the timestamp	(*Unit: seconds*)
# * **relative_humidity:**	relative humidity measured at the timestamp	(*Unit: percent*)

# In[ ]:


data.shape


# In[ ]:


data.head()


# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold"><br>
# 
# Data Sampling<br></p>
# 
# Lots of rows, so let us sample down by taking every 10th row. <br>
# 

# In[ ]:


sampled_df = data[(data['rowID'] % 10) == 0]
sampled_df.shape


# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold"><br>
# 
# Statistics
# <br><br></p>
# 

# In[ ]:


sampled_df.describe().transpose()


# In[ ]:


sampled_df[sampled_df['rain_accumulation'] == 0].shape


# In[ ]:


sampled_df[sampled_df['rain_duration'] == 0].shape


# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold"><br>
# 
# Drop all the Rows with Empty rain_duration and rain_accumulation
# <br><br></p>
# 

# In[ ]:


del sampled_df['rain_accumulation']
del sampled_df['rain_duration']


# In[ ]:


rows_before = sampled_df.shape[0]
sampled_df = sampled_df.dropna()
rows_after = sampled_df.shape[0]


# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold"><br>
# 
# How many rows did we drop ?
# <br><br></p>
# 

# In[ ]:


rows_before - rows_after


# In[ ]:


sampled_df.columns


# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold"><br>
# 
# Select Features of Interest for Clustering
# <br><br></p>
# 

# In[ ]:


features = ['air_pressure', 'air_temp', 'avg_wind_direction', 'avg_wind_speed', 'max_wind_direction', 
        'max_wind_speed','relative_humidity']


# In[ ]:


select_df = sampled_df[features]


# In[ ]:


select_df.columns


# In[ ]:


select_df


# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold"><br>
# 
# Scale the Features using StandardScaler
# <br><br></p>
# 

# In[ ]:


X = StandardScaler().fit_transform(select_df)
X


# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold"><br>
# 
# Use k-Means Clustering
# <br><br></p>
# 

# In[ ]:


kmeans = KMeans(n_clusters=12)
model = kmeans.fit(X)
print("model\n", model)


# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold"><br>
# 
# What are the centers of 12 clusters we formed ?
# <br><br></p>
# 

# In[ ]:


centers = model.cluster_centers_
centers


# <p style="font-family: Arial; font-size:2.75em;color:purple; font-style:bold"><br>
# 
# Plots
# <br><br></p>
# 

# Let us first create some utility functions which will help us in plotting graphs:

# In[ ]:


# Function that creates a DataFrame with a column for Cluster Number

def pd_centers(featuresUsed, centers):
	colNames = list(featuresUsed)
	colNames.append('prediction')

	# Zip with a column called 'prediction' (index)
	Z = [np.append(A, index) for index, A in enumerate(centers)]

	# Convert to pandas data frame for plotting
	P = pd.DataFrame(Z, columns=colNames)
	P['prediction'] = P['prediction'].astype(int)
	return P


# In[ ]:


# Function that creates Parallel Plots

def parallel_plot(data):
	my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(data)))
	plt.figure(figsize=(15,8)).gca().axes.set_ylim([-3,+3])
	parallel_coordinates(data, 'prediction', color = my_colors, marker='o')


# In[ ]:


P = pd_centers(features, centers)
P


# # Dry Days

# In[ ]:


parallel_plot(P[P['relative_humidity'] < -0.5])


# # Warm Days

# In[ ]:


parallel_plot(P[P['air_temp'] > 0.5])


# # Cool Days

# In[ ]:


parallel_plot(P[(P['relative_humidity'] > 0.5) & (P['air_temp'] < 0.5)])

