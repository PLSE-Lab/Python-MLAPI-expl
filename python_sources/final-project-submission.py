#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This dataset contains information regarding traffic congestion in major US Interstates and attempts to create a model that accurately predicts future congestion. The development of this notebook will take on the following structure:
# 
# * Exploratory Analysis: This stage will explore the data, rename labels as appropriate and discover that kind of pre-processing must be made (whether there are empty datapoints, distribution of data, entropy of each feature). 
# * Preprocessing: This stage will pre-process the data to put it in a way the model can make an accurate prediction.
# * Algorithm selection and implementation
# * Model evaluation and optimization stage
# * Custom model creation (optional)

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import folium
from folium.plugins import HeatMap

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# Setting up BigQuery library
PROJECT_ID = 'villanova-project'
from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID, location="US")
from google.cloud import storage
storage_client = storage.Client(project=PROJECT_ID)
from google.cloud import automl_v1beta1 as automl
automl_client = automl.AutoMlClient()
from google.cloud.bigquery import magics
from kaggle.gcp import KaggleKernelCredentials
magics.context.credentials = KaggleKernelCredentials()
magics.context.project = PROJECT_ID

# load biquery commands
get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')


# # Exploratory Analysis

# In[ ]:


# Read data
df_train = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/train.csv')
df_test = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/test.csv')


# First thing to check is the data type of each column:

# In[ ]:


df_train.info()


# Our test data however, has a different shape to our training data: 

# In[ ]:


df_test.info()


# Training data has 27 features in total while testing data has 13. Some are labeled as objects since they are string data while the rest are numerical data. Next, let's see which, and how much, data is missing. We'll start with numerical data:

# In[ ]:


df_train.isnull().sum()


# There are only two numerical features that have nan values. We'll have to fill in these values later in the processing stage. Now let's see how much categorical data is missing:
# 

# In[ ]:


obj_df = df_train.select_dtypes(include=['object'])
obj_df[obj_df.isnull().any(axis=1)].count()


# Looks like EntryHeading, EntryHeading, Path, City and ExitStreetName have the same amount of null values. The amount for each of these values  represents about **1.5%** of our training data for that column. This seems like a small slice of our data but we can't gurantee that ignoring these rows won't affect the final outcome. We'll try to fill in these values later as well.
# 
# To gain a better understanding of the data, let's see what the distirbution is for each of the features. We will start with count for cities since that is the easiest to digest.

# In[ ]:


# sns.pairplot(df_train)


# In[ ]:


# Checking for distribution of ALL DATA for each city
train_plot = sns.countplot(x="City", data=df_train)
train_plot


# We can clearly see that Philadelphia has the highest data count of all cities. The data is therefore unevenly distributed and this could affect our models. However, this is total count for all cities and that doesn't yield too much more information, let's group the datacount by the number of unique Intersection Ids - this is the id given to each intersection where traffic data is being measured.

# In[ ]:


# Checking for distribution of data BY UNIQUE INTERSECTION ID
fig = df_train.groupby(['City'])['IntersectionId'].nunique().sort_index().plot.bar()
fig.set_title('# of Intersections per city in train Set', fontsize=15)
fig.set_ylabel('# of Intersections', fontsize=15);
fig.set_xlabel('City', fontsize=17);


# Interesting, altough Philadelphia has the highest data count, Chicago has the highest number of unique intersections. This tells us that, although Philly has more data, Chicago has the larger share of unique intersections. Does this mean that Chicago has more traffic that Philly? Not necessarily, Chicago can have more intersections but Philadelphia can have more traffic total. 
# 
# To explore this assumption, let's find out what is the distribution of the traffic by month and week:

# In[ ]:


# let's see the distribution of traffic by month and date
plt.figure(figsize=(15,12))

plt.subplot(211)
g = sns.countplot(x="Hour", data=df_train, hue='City', dodge=True)
g.set_title("Distribution by hour and city", fontsize=20)
g.set_ylabel("Count",fontsize= 17)
g.set_xlabel("Hours of Day", fontsize=17)
sizes=[]
for p in g.patches:
    height = p.get_height()
    sizes.append(height)

g.set_ylim(0, max(sizes) * 1.15)

plt.subplot(212)
g1 = sns.countplot(x="Month", data=df_train, hue='City', dodge=True)
g1.set_title("Hour Count Distribution by Month and City", fontsize=20)
g1.set_ylabel("Count",fontsize= 17)
g1.set_xlabel("Months", fontsize=17)
sizes=[]
for p in g1.patches:
    height = p.get_height()
    sizes.append(height)

g1.set_ylim(0, max(sizes) * 1.15)

plt.subplots_adjust(hspace = 0.3)

plt.show()


# Again, philly comes out on top when it comes to count of traffic data. However this still doesn't give enough support to the theory that Philly has more traffic simply because it has more data. We could check this assumption by seeing how much actual stopping there is in philly traffic vs other city's traffic.
# 

# In[ ]:




fig = df_train.groupby(['City']).TotalTimeStopped_p80.median().sort_index().plot(kind='barh')

fig.set_title('Average Stopping Time', fontsize=15)
fig.set_ylabel('City', fontsize=10);
fig.set_xlabel('Minutes stopped at intersection', fontsize=10);


# We can see that Atlanta actually has more stopping time than any other city, even Philly. Even though Atlanta comes 3rd in number of datapoints and last in number of intersections. The number of intersections therefore is not necessarily the most important factor in determining traffic. 
# 
# If that's the case, then the properties of the intersection itself for each city will need to be investigated more. Let's explore further the congestion for each city. We are given in this dataset the street (Path) as well as the entryway and exitway direction. If we can see where the places are of largest congestion for each city, we could potentially associate the traffic to a particular intersection. Furthermore, if we think about what kind of things cause traffic like weather, schools, sports stadiums, closeness to city center, etc. then we could associate those places of high congestion to one of these factors. But enough speculation let's dive to the data.

# Let's create a unique dataframe for each city.

# In[ ]:


Atlanda=df_train[df_train['City']=='Atlanta'].copy()
Boston=df_train[df_train['City']=='Boston'].copy()
Chicago=df_train[df_train['City']=='Chicago'].copy()
Philadelphia=df_train[df_train['City']=='Philadelphia'].copy()


# Now lets create column, TotalTimeToWait, that adds up all the waiting times

# In[ ]:


Atlanda['TotalTimeWaited']=Atlanda['TotalTimeStopped_p20']+Atlanda['TotalTimeStopped_p40']+Atlanda['TotalTimeStopped_p50']+Atlanda['TotalTimeStopped_p60']+Atlanda['TotalTimeStopped_p80']
Boston['TotalTimeWaited']=Boston['TotalTimeStopped_p20']+Boston['TotalTimeStopped_p40']+Boston['TotalTimeStopped_p50']+Boston['TotalTimeStopped_p60']+Boston['TotalTimeStopped_p80']
Chicago['TotalTimeWaited']=Chicago['TotalTimeStopped_p20']+Chicago['TotalTimeStopped_p40']+Chicago['TotalTimeStopped_p50']+Chicago['TotalTimeStopped_p60']+Chicago['TotalTimeStopped_p80']
Philadelphia['TotalTimeWaited']=Philadelphia['TotalTimeStopped_p20']+Philadelphia['TotalTimeStopped_p40']+Philadelphia['TotalTimeStopped_p50']+Philadelphia['TotalTimeStopped_p60']+Philadelphia['TotalTimeStopped_p80']


# In[ ]:


Atlanda['TotalTimeWaited'].hist(bins=100)


# Now let's plot the highest waiting times for each street (Path).

# In[ ]:


temp_1=Atlanda.groupby('EntryStreetName')['TotalTimeWaited'].mean().sort_values().tail(10)
temp_1.plot(kind='barh',title='Highest traffic startng street in Atlanta')


# In[ ]:


temp_2=Boston.groupby('EntryStreetName')['TotalTimeWaited'].mean().sort_values().tail(10)
temp_2.plot(kind='barh',title='Highest traffic startng street in Boston')


# In[ ]:


temp_3=Chicago.groupby('EntryStreetName')['TotalTimeWaited'].mean().sort_values().tail(10)
temp_3.plot(kind='barh',title='Highest traffic startng street in Chicago')


# In[ ]:


temp_4=Philadelphia.groupby('EntryStreetName')['TotalTimeWaited'].mean().sort_values().tail(10)
temp_4.plot(kind='barh',title='Highest traffic startng street in Philadelphia')


# ## Map Analysis

# Let's plot out the intersections for each city using the latitude and longitude in the dataset

# In[ ]:


# use plotly to plot where intersections are for all cities. Provide observational data. 
# then do a heatmap which groups intersectionId with TotalStoppingTime across space to see where the heaviest traffic is. 
# Investigate what's around here and provide observations.
traffic_df=Atlanda.groupby(['Latitude','Longitude'])['TotalTimeWaited'].count().to_frame()
traffic_df.columns.values[0]='count1'
traffic_df=traffic_df.reset_index()
lats=traffic_df[['Latitude','Longitude','count1']].values.tolist()
    
hmap = folium.Map(location=[33.7638493,-84.3801108], zoom_start=12)
hmap.add_child(HeatMap(lats, radius = 6))
hmap


# In[ ]:


traffic_df=Boston.groupby(['Latitude','Longitude'])['TotalTimeWaited'].count().to_frame()
traffic_df.columns.values[0]='count1'
traffic_df=traffic_df.reset_index()
lats=traffic_df[['Latitude','Longitude','count1']].values.tolist()
    
hmap = folium.Map(location=[42.3158246,-71.0787574], zoom_start=12)
hmap.add_child(HeatMap(lats, radius = 6))
hmap


# In[ ]:


traffic_df=Chicago.groupby(['Latitude','Longitude'])['TotalTimeWaited'].count().to_frame()
traffic_df.columns.values[0]='count1'
traffic_df=traffic_df.reset_index()
lats=traffic_df[['Latitude','Longitude','count1']].values.tolist()
    
hmap = folium.Map(location=[41.8420892,-87.7237629], zoom_start=11)
hmap.add_child(HeatMap(lats, radius = 6))
hmap


# In[ ]:


traffic_df=Philadelphia.groupby(['Latitude','Longitude'])['TotalTimeWaited'].count().to_frame()
traffic_df.columns.values[0]='count1'
traffic_df=traffic_df.reset_index()
lats=traffic_df[['Latitude','Longitude','count1']].values.tolist()
    
hmap = folium.Map(location=[39.9484792,-75.1774329], zoom_start=12)
hmap.add_child(HeatMap(lats, radius = 6))
hmap


# The maps show a clear correlation between congestion and distance to city center. If we can come up with a "weight" for this observation then we can more intelligently fill missing values.
# 
# There's another correlation that we can explore and that is weather.

# ## Correlation between features
# To figure out the correlations between features, we can draw out a heatmap. Then, we can use PCA analysis to redeuce the dimensionality of the features to see how closely thay are related with one another from another perspective.

# In[ ]:


t_stopped = ['TotalTimeStopped_p20',
             'TotalTimeStopped_p50', 
             'TotalTimeStopped_p80']
t_first_stopped = ['TimeFromFirstStop_p20',
                   'TimeFromFirstStop_p50',
                   'TimeFromFirstStop_p80']
d_first_stopped = ['DistanceToFirstStop_p20',
                   'DistanceToFirstStop_p50',
                   'DistanceToFirstStop_p80']


# In[ ]:


plt.figure(figsize=(15,12))
plt.title('Correlation of Time and Distance Stopped', fontsize=17)
sns.heatmap(df_train[t_stopped + t_first_stopped + d_first_stopped].astype(float).corr(), vmax=1.0,  annot=True)
plt.show()


# We can see that there's high correlation between TimeFromFirstStop_p20,TimeFromFirstStop_50,TimeFromFirstStop_p80 and TotalTimeStopped_p20, TotalTimeStopped_p50, TotalTimeStopped_p80. There is not a lot of correlation between TotalTimeStopped_p20 and DistanceToFirstStop80 and vice versa.

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(12,5))

ax[0].set_title('Total time stopped vs Time from first stop', fontsize=15)
sns.scatterplot(x="TotalTimeStopped_p80", y='TimeFromFirstStop_p80', data=df_train, ax=ax[0])

ax[1].set_title('Total time stopped vs Distance to first stop', fontsize=15)
sns.scatterplot(x="TotalTimeStopped_p80", y='DistanceToFirstStop_p80', data=df_train, ax=ax[1])

plt.show()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(12,5))

ax[0].set_title('Total time stopped vs Distance to first stop', fontsize=15)
sns.scatterplot(x="TimeFromFirstStop_p80", y='DistanceToFirstStop_p80', data=df_train, ax=ax[0])

ax[1].set_title('Total time stopped per IntersectionId', fontsize=15)
sns.scatterplot(x="TotalTimeStopped_p80", y='IntersectionId', data=df_train, ax=ax[1])

plt.show()


# # Preprocessing

# In[ ]:


df_train_cleaned = df_train.copy().dropna()


# In[ ]:


df_train_cleaned['TotalTimeStopped_p20'].hist()


# In[ ]:


# totalTimeStopped['TotalTimeStopped_p20'] = np.log(totalTimeStopped['TotalTimeStopped_p20'])
# totalTimeStopped.hist('TotalTimeStopped_p20',figsize=(8,5))


# # Algorithm Selection and Run
# 
# The following SQL query is made to create a logistical regression model using BigQuery's ML API:
# ```sql
# CREATE MODEL IF NOT EXISTS `bqml_dataset.model_TotalTimeStopped_p20`
# OPTIONS(model_type='linear_reg') AS
# SELECT
#     TotalTimeStopped_p20 as label,
#     Weekend,
#     Hour,
#     EntryStreetName,
#     ExitStreetName,
#     Path,
#     EntryHeading,
#     ExitHeading,
#     City,
#     Month
# FROM
#   `bqml_dataset.cleaned_training_data`
# WHERE
#     RowId < 2600000
# ```
# I create a linear regression model, `bqml_dataset.model1`, using labels that are present in both training and testing data. In this case, TotalTimeStopped_p20 is the label for which I want to run the regression. This model does not include IntersectionId, Latitude or Longitude, 
# 
# **Note**: this is inputted as markdown for pricing considerations when running this notebook.

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM\n  ML.TRAINING_INFO(MODEL `bqml_dataset.model_TotalTimeStopped_p20`)\nORDER BY iteration ')


# The following SQL query is then run to run a prediction using testing data: 
# ```
# SELECT
#   predicted_label AS TotalTimeStopped_p20
# FROM
#   ML.PREDICT(MODEL `bqml_dataset.model_TotalTimeStopped_p20`,
#     (
#     SELECT
#       Weekend,
#       Hour,
#       EntryStreetName,
#       ExitStreetName,
#       Path,
#       EntryHeading,
#       ExitHeading,
#       City,
#       Month
#     FROM
#       `kaggle-competition-datasets.geotab_intersection_congestion.test`))
# ```
# 
# The results of this model's prediction are provided by BigQuery:
# * Mean absolute error: 2.8594
# * Mean squared error: 45.1412
# * Mean squared log error: 1.0558
# * Median absolute error: 1.2165
# * R squared: 0.1939
# 
# The low absolute error tells us that the predictions overall were pretty close to the actual values. The fact that the mean square error is much higher (45) tells us that there were some predicted values that were off from the predicted value by a significant amount. 
# 
# As I created models for the other features, I noticed that the mean squared error -the value for which my predictions are scored in the competition- kept getting higher. Below are some of the scores for Total Time Stopped:
# 
# TotalTimeStopped_p20
# * Mean absolute error: 2.9238
# * Mean squared error: 42.8358
# 
# TotalTimeStopped_p50
# * Mean absolute error: 7.8873
# * Mean squared error: 163.8435
# 
# TotalTimeStopped_p80
# * Mean absolute error: 14.9579
# * Mean squared error: 491.1482
# 
# At this point I had to concede that there were some values that were way off from the mark. Trying to figure out which data was causing this would be very difficult, so I instead resorted to using another algorithm to make the predictions. 

# In[ ]:


# get table prediction
# create a table. one column has predicted values, the other actual.
# sort the table by highest difference
# analze the top datapoints for any common patterns


# ### Commentary:
# The dataset leans heavily towards Philadelphia, which can skew the result of our model. From the hour and month plots we can draw some observations:
# * There is less data in the early hours of the morning for all cities, increasing throughout the day
# * Philadelphia data count peaks between 3pm-7pm
# * Boston peaks at 10am and gradually falls
# * Chicago closely follow Boston's trend throughout the day
# * Atlanta stays constant throughout the day starting around 7-8am
# 
# When it comes to months of the year:
# * There is significantly less data in the spring
# * Data count for Boston increases towards the end of the year
# * Philadelphia has the highest data count of any other city
# 
# Latitude and Longitude can help pinpoint us where exactly the vehicle was going at the time the data was captured. 'EntryStreetName' and 'ExitStreetName' provide directional data, which can tell us on *what* direction traffic is flowing.  Perhaps traffic is heavier on certain streets only in one direction rather than the other (think commuters).
# 
# 'Month', 'Weekend' and 'Day' are really interesting because they provides us a chance to do analysis across time. Seasons change, which can bring more congestion (think of how much traffic slows during snowstorms as opposed to a sunny day). School breaks influence traffic, so do important holidays and recurring cultural events.

# # Aknowledgements:
# I would like to thank Professor Mitchell for advising me in this project and his enthusiasm for data science. I would also like to aknowledge the following Kaggle users who provided inspiration in the analysis of the data: whatust, Faith Bilgin, Leonardo Ferreira and Pradeep Muniasamy
