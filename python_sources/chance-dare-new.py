#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


# Imports
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
get_ipython().system('pip install category_encoders ### Uncomment this went running notebook for the first time ###')
import category_encoders as ce


# # Bring in the CSV's!

# In[ ]:


# Create the dataframes from the csv's
train = pd.read_csv('../input/train_features.csv')
test = pd.read_csv('../input/test_features.csv')
train_labels = pd.read_csv('../input/train_labels.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')


# # Grab the test ids for later

# In[ ]:


test_ids = test['id']


# # Wrangle code with this function(Thanks Ryan Herr)

# In[ ]:


def wrangle(X):
    """Wrangles train, validate, and test sets in the same way"""
    X = X.copy()

    # Convert date_recorded to datetime
    X['date_recorded'] = pd.to_datetime(X['date_recorded'], infer_datetime_format=True)
    
    # Extract components from date_recorded, then drop the original column
    X['year_recorded'] = X['date_recorded'].dt.year
    X['month_recorded'] = X['date_recorded'].dt.month
    X['day_recorded'] = X['date_recorded'].dt.day
    X = X.drop(columns='date_recorded')
    
    # Engineer feature: how many years from construction_year to date_recorded
    X['years'] = X['year_recorded'] - X['construction_year']    
    
    # Drop recorded_by (never varies) and id (always varies, random)
    unusable_variance = ['recorded_by', 'id', 'num_private', 
                         'wpt_name', 'permit', 'management_group',
                         'water_quality', 'year_recorded',
                         'extraction_type_group']
    X = X.drop(columns=unusable_variance)
    
    # Drop duplicate columns
    duplicate_columns = ['quantity_group']
    X = X.drop(columns=duplicate_columns)
    
    # About 3% of the time, latitude has small values near zero,
    # outside Tanzania, so we'll treat these like null values
    X['latitude'] = X['latitude'].replace(-2e-08, np.nan)
    
    # When columns have zeros and shouldn't, they are like null values
    cols_with_zeros = ['construction_year', 'longitude', 'latitude',
                       'gps_height', 'population']
    for col in cols_with_zeros:
        X[col] = X[col].replace(0, np.nan)
        
    return X


# # Gonna slap the labels onto my training df

# In[ ]:


train['status_group'] = train_labels['status_group']


# # Clustering with KMeans! Pretty fancy graphs if I do say so myself!

# In[ ]:


from sklearn.cluster import KMeans
train['longitude'].values
points = train[['longitude', 'latitude']].fillna(0).values
kmeans = KMeans(n_clusters=14, random_state=42)
kmeans.fit(points)
y_km = kmeans.fit_predict(points)

data = pd.DataFrame({'lon': points[:,0], 'lat': points[:,1], 'cluster':y_km})

train['cluster'] = data['cluster']

# color = data['cluster'].replace({0:'r', 1:'orange', 2:'y', 3:'g',
#                                  4:'b', 5:'indigo', 6:'violet', 7:'black',
#                                  8:'lightblue', 9:'lightgreen', 10:'grey', 
#                                  11:'r', 12:'brown', 13:'brown', 14:'purple',
#                                  15:'black'})


# plt.figure(figsize=(10, 10))
# plt.style.use('ggplot')
# plt.scatter(train['longitude'], train['latitude'], s=2, c = color)
# plt.xlabel('Longitude', fontsize=20)
# plt.ylabel('Latitude', fontsize=20)


# # Let's go ahead and cluster those testy bois too

# In[ ]:


test['longitude'].values
points = test[['longitude', 'latitude']].fillna(0).values
kmeans = KMeans(n_clusters=14)
kmeans.fit(points)
y_km = kmeans.fit_predict(points)

data = pd.DataFrame({'lon': points[:,0], 'lat': points[:,1], 'cluster':y_km})

test['cluster'] = data['cluster']


# # Train Validation split!

# In[ ]:


train, val = train_test_split(train, test_size= test.shape[0], stratify=train['status_group'])


# # Use that wrangle function from earlier on the train, validation, and test dataframes! (Thanks again Ryan)

# In[ ]:


train = wrangle(train)
val = wrangle(val)
test = wrangle(test)


# # Create an X_train, X_val, y_train, y_val, and X_test

# In[ ]:


X_train = train.drop('status_group', axis=1)
y_train = train['status_group']
X_val = val.drop('status_group', axis=1)
y_val = val['status_group']
X_test = test


# # Create pipeline and fit with Random Forest Classifier, can't forget the SimpleImputer and OrdinalEncoder!

# In[ ]:


rfc = RandomForestClassifier(n_estimators=1000,
                               min_samples_split=6,
                               max_depth = 23,
                               criterion='gini',
                               max_features='auto',
                               random_state=42,
                               n_jobs=-1)

pipe = make_pipeline(
    ce.OrdinalEncoder(),
    SimpleImputer(strategy='median'),
    rfc)

pipe.fit(X_train, y_train)
pipe.score(X_train, y_train)


# # Some fancy predictions

# In[ ]:


y_pred = pipe.predict(X_val)
accuracy_score(y_pred, y_val)


# # Might as well graph these importances and add an arbitrary vline

# In[ ]:


importances = rfc.feature_importances_
features = X_train.columns
plt.style.use('ggplot')
plt.figure(figsize=(10, 10))
plt.barh(features, importances)
plt.axvline(.008, c='b')


# # Turns out I like the results, let's predict the test features now!

# In[ ]:


y_pred = pipe.predict(X_test)


# In[ ]:


y_pred = pipe.predict(X_test)
sub = pd.DataFrame(data = {
    'id': test_ids,
    'status_group': y_pred
})
sub.to_csv('submission.csv', index=False)


# # Here's another visualization for the fun of it.

# In[ ]:


color = train['status_group'].replace({'functional': 'g', 'functional needs repair': 'y', 'non functional': 'r'})
plt.figure(figsize=(10, 10))
plt.style.use('ggplot')
plt.scatter(train['longitude'], train['latitude'], s=2, color = color)
plt.xlabel('Longitude', fontsize=20)
plt.ylabel('Latitude', fontsize=20)
red_patch = mpatches.Patch(color='red', label='Non-Functional')
green_patch = mpatches.Patch(color='green', label='Functional')
yellow_patch = mpatches.Patch(color='yellow', label='Functional, Needs Work')
plt.legend(handles=[green_patch, yellow_patch, red_patch])


# In[ ]:


train.head()


# In[ ]:


plt.scatter(train['population'], train['quantity'], alpha=.1)


# In[ ]:




