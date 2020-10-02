#!/usr/bin/env python
# coding: utf-8

# # Weather forecast
# We will predict upcomming events for current day by feeding the model with data about
# * Temperature
# * Dew point temperature
# * Humidity percentage
# * Sea level pressure
# * Visibility
# * Wind speed
# * Percipitation

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# ### Get data (X) and events (y) from CSV file

# In[2]:


df = pd.read_csv('../input/austin_weather.csv')
df.set_index('Date').sort_index()

# use average data only
columns_of_interest = ['TempAvgF','DewPointAvgF', 'HumidityAvgPercent', 'SeaLevelPressureAvgInches', 'VisibilityAvgMiles', 'WindAvgMPH', 'PrecipitationSumInches']
data = df[columns_of_interest]
events = df[['Events']].replace(' ', 'None')


# ## Data exploration
# 
# 
# Let's see what lies in column that will become our prediction / source of truth for training:
# ### Plot event categories

# In[3]:


events.Events.value_counts().plot(kind='bar', figsize=(10,5))


# We have information what events occurred for given weather parameters. 
# 
# We can see that single column combines multiple events.
# 
# Separating them seems to be a good idea as it will allow us to predict all of the events combinations independently.
# 
# Let's see what unique categories do we have in Events column:

# ### Get unique events categories

# In[4]:


unique_events = set()
for value in events.Events.value_counts().index:
    splitted = [x.strip() for x in value.split(',')]
    unique_events.update(splitted)
unique_events


# Now we can try to generate histogram of independent event. 
# 
# We will see that "Snow" occurred only once.
# 
# ### Print histogram with single events only

# In[5]:


single_events = pd.DataFrame()
for event_type in unique_events:
    event_occurred = events.Events.str.contains(event_type)
    single_events = pd.concat([single_events, pd.DataFrame(data={event_type: event_occurred.values})], join='outer', axis=1)

#single_events.head()
ax = single_events.sum().sort_values(ascending=False).plot.bar(figsize=(10,5))
ax.set_title("Weather events in dataset", fontsize=18)
ax.set_ylabel("Number of occurrences", fontsize=14)
for i in ax.patches:
    ax.text(i.get_x()+.18, i.get_height()+5, i.get_height(), fontsize=12)


# At this point we also have a nice table of predictions with events separated:

# In[6]:


single_events.head()


# During exploration I noticed that PrecipitationSumInches colum has mysterious "T" values which means "Trace". It means that there was a trace of rain but it was not measureable.
# 
# Let's check if all other values in this column are numbers or do we have anything else there:
# 
# ### Check how many traces do we have in PrecipitationSumInches collumn

# In[7]:


precipitation = data[pd.to_numeric(data.PrecipitationSumInches, errors='coerce').isnull()].PrecipitationSumInches.value_counts()
precipitation


# Let's check rest of the columns with non-numeric values:
# ### Find all non numeric rows in data frame

# In[8]:


# this function returns array with one item for each row
# each item indicates if the row with columns of our interest had non-numeric data
def isColumnNotNumeric(columns_of_interest, data):
    result = np.zeros(data.shape[0], dtype=bool)
    for column_name in columns_of_interest:
        result = result | pd.to_numeric(data[column_name], errors='coerce').isnull()
    return result

def getDataFrameWithNonNumericRows(dataFrame):
    return data[isColumnNotNumeric(columns_of_interest, data)]

non_numeric_rows_count = getDataFrameWithNonNumericRows(data).shape[0]

print("Non numeric rows: {0}".format(non_numeric_rows_count))


# ## Data transformations
# 
# ### Replace "Trace" values in PrecipitationSumInches with 0 and add another column PercipitationTrace
# This new column will get values 0 if there was no trace of precipitation and 1 if there was a trace

# In[9]:


def numberOrZero(value):
    try:
        parsed = float(value)
        return parsed
    except:
        return 0
# this line is unnecessary if we run script from top to bottom, 
# but it helps debugging this part of code to get fresh PrecipitationSumInches column
data['PrecipitationSumInches'] = df['PrecipitationSumInches']

#Find rows indices with "T" values
has_precipitation_trace_series = isColumnNotNumeric(['PrecipitationSumInches'], data).astype(int)
#data['PrecipitationTrace'] = has_precipitation_trace_series
#data.loc[:,'PrecipitationTrace'] = has_precipitation_trace_series
data = data.assign(PrecipitationTrace=has_precipitation_trace_series.values)

data['PrecipitationSumInches'] = data['PrecipitationSumInches'].apply(numberOrZero)
data.iloc[0:10,:]


# Check how many non numeric rows we still have

# In[10]:


getDataFrameWithNonNumericRows(data)


# As there are not so many missing values, we can drop missing data.
# 
# We need to get rows indices first to drop them as well in events table
# ### Drop rows with missing values

# In[11]:


row_indices_for_missing_values = getDataFrameWithNonNumericRows(data).index.values
row_indices_for_missing_values
data_prepared = data.drop(row_indices_for_missing_values)
events_prepared = single_events.drop(row_indices_for_missing_values)
print("Data rows: {0}, Events rows: {1}".format(data_prepared.shape[0], events_prepared.shape[0]))


# ### Convert dataframe columns to be treated as numbers

# In[12]:


data_prepared.dtypes


# In[13]:


data_prepared = data_prepared.apply(pd.to_numeric)
data_prepared.dtypes


# ### Normalize input data
# 

# In[14]:


from sklearn import preprocessing
data_values = data_prepared.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()

data_prepared = pd.DataFrame(min_max_scaler.fit_transform(data_prepared), columns=data_prepared.columns, index=data_prepared.index)


# ## Final look at the  prepared data

# In[15]:


data_prepared.head()


# In[16]:


events_prepared.head()


# ## Train the model
# ### Split the data into train and test sets

# In[17]:


from sklearn.model_selection import train_test_split

random_state = 42
X_train, X_test = train_test_split(data_prepared, test_size=0.2, random_state=random_state)
y_train, y_test = train_test_split(events_prepared, test_size=0.2, random_state=random_state)

clusters_count = len(unique_events)


# ### Check if we can figure out events by discovering them using unsupervised learning techniques

# In[18]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=clusters_count).fit(X_train)

resultDf = pd.DataFrame(kmeans.labels_)
resultDf.iloc[:,0].value_counts().plot.bar()
#plt.hist(kmeans.labels_)


# In[19]:


from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=clusters_count, linkage="average").fit(X_train)
resultDf = pd.DataFrame(ac.labels_)
resultDf.iloc[:,0].value_counts().plot.bar()


# ### Ground truth

# In[20]:


events_prepared.sum().sort_values(ascending=False).plot.bar()


# As we can see AgglomerativeClustering did some nice work representing similar distribution of categories as for real data.
# 
# However it can output only single event so we are unable to know that there will be Rain and Fog at the same time.
# 
# ## Multi-label clustering
# 
# We need to calculate cluster centers. That way, for given new sample, we can measure distance from each cluster. Using some distance threshold we will be able to assign new data to one or more clusters.
# 
# Let's have a look at these 2 charts:

# In[21]:


fig, ax = plt.subplots(1, 2, figsize=(15, 5))
events_prepared.sum().sort_values(ascending=False).plot.bar(ax=ax[0], title="Real events that happened")
resultDf.iloc[:,0].value_counts().plot.bar(ax=ax[1], title="Histogram obtained from agglomerative clustering")


# We can now try to map cluster numbers to category names.

# In[22]:


event_names_ordered = events_prepared.sum().sort_values(ascending=False).index
clusters_ordered = resultDf.iloc[:,0].value_counts().index
cluster_category_mapping = {}
for i in range(clusters_count):
    cluster_category_mapping.update({clusters_ordered[i]:event_names_ordered[i]})
cluster_category_mapping


# ## Find clusters centroids to classify test data
# We need to have a way to classify new data on already trained clusters.
# 
# We can do this by calculating clusters center and measure distance from each center 

# In[23]:


cluster_centers_mapping = {}
for key in cluster_category_mapping:
    cluster_indices = resultDf.loc[resultDf[0] == key].index
    cluster_data = X_train.iloc[cluster_indices]
    mean = cluster_data.mean(axis=0).values
    #print("\n" + cluster_category_mapping[key])
    #print(mean)
    cluster_centers_mapping.update({key:mean})
cluster_centers_mapping


# We now need to calculate distances between these centroids and our data points

# In[24]:


def get_distances_from_cluster(data_frame):
    cluster_distance = np.zeros((data_frame.shape[0], clusters_count))
    #cluster_distance
    for i in range(data_frame.shape[0]):
        #print(X_test.iloc[[i]].values[0])
        for key in cluster_category_mapping:
            dist = np.linalg.norm(data_frame.iloc[[i]].values[0]-cluster_centers_mapping[key])
            cluster_distance[i,key] = dist
            #print(dist)
    column_names = [cluster_category_mapping[k] for k in cluster_category_mapping]
    #column_names

    return pd.DataFrame(cluster_distance, index=data_frame.index, columns=column_names)

distancesDf = get_distances_from_cluster(X_train)
distancesDf.head()


# Column with minimum distance is our classification. But to have ability to classify a record to both clusters, we can add some tolerance. 
# For example 
# > Data point belongs to cluster if distance from cluster is smaller than minimum distance * 1.2

# In[25]:


def classify_events(distances_dataFrame):
    return distances_dataFrame.apply(lambda x: x<x.min()*1.02, axis=1)

classification_result = classify_events(distancesDf)
X_train_col_ordered = classification_result.reindex(sorted(classification_result.columns), axis=1)
y_train_col_ordered = y_train.reindex(sorted(y_train.columns), axis=1)

#check if all columns and all rows are equal in both datasets
def check_accuracy(X, y):    
    comparison = X == y

    val_counts = comparison.all(axis=1).value_counts()
    percentageCorrect = val_counts.at[True] / X.shape[0] * 100
    return percentageCorrect
    


# In[26]:


check_accuracy(X_train_col_ordered, y_train_col_ordered)


# In[27]:


X_train_col_ordered.head()


# In[28]:


y_train_col_ordered.head()


# ## Predicted weather events based on agglomerative clustering with unsupervised learning

# In[29]:


fig, ax = plt.subplots(1, 2, figsize=(15, 5))
y_train_col_ordered.sum().plot.bar(ax=ax[0], title="Real events that happened")
X_train_col_ordered.sum().plot.bar(ax=ax[1], title="Predicted events")
#resultDf.iloc[:,0].value_counts().plot.bar(ax=ax[1], title="Histogram obtained from agglomerative clustering")


# In[30]:


distancesDf = get_distances_from_cluster(X_test)
classification_result = classify_events(distancesDf)
X_test_col_ordered = classification_result.reindex(sorted(classification_result.columns), axis=1)
y_test_col_ordered = y_test.reindex(sorted(y_train.columns), axis=1)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
y_test_col_ordered.sum().plot.bar(ax=ax[0], title="Real events that happened")
X_test_col_ordered.sum().plot.bar(ax=ax[1], title="Predicted events")


# In[31]:


check_accuracy(X_test_col_ordered, y_test_col_ordered)


# In[ ]:




