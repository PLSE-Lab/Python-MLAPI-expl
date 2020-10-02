#!/usr/bin/env python
# coding: utf-8

# # Predicting and Analysing San Francisco Crimes

# This notebook attempts to analysis and predict the class of crimes committed within the city of San Francisco. The code first exploring and visualising crime patterns across the neighbourhoods and police districts.
# 
# Then applying a machine learning algorithm in order to guess the category of the crimes based on their time and location of occurrence by using the Naive Bayes classifier as it's one of the simplest classification algorithms.

# In[ ]:


# load all the needed packages
import time
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder
from shapely.geometry import Point, Polygon, shape
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


sns.set(style = 'darkgrid')
sns.set_palette('PuBuGn_d')


# In[ ]:


# load dataset then parsing the dates column into datetime
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


# show the train data
train.head(3)


# In[ ]:


# show the tolal number of recoreds
train.shape


# In[ ]:


# check for NAN values
train.isnull().sum()


# So there are no missing values in the dataset.

# In[ ]:


# check for doublications
train.duplicated().any()


# In[ ]:


# show the number of doublications
train.duplicated().sum()


# Now 2323 is a small number compared to the total number of rows which means we can drop them!

# In[ ]:


# drop duplicate rows and keep only the unique values
train = train.drop_duplicates()
train.shape


# In[ ]:


# explore crime categories
train['Category'].unique()


# In[ ]:


# plot the total number of incidents for each category
x = sns.catplot('Category', data = train, kind = 'count', aspect = 3, height = 4.5)
x.set_xticklabels(rotation = 85)


# For more visualisation you need to check out this [notebook](https://nbviewer.jupyter.org/github/just4data/San-Francisco-Crime-Analysis-And-Prediction/blob/master/sf_crime_classification.ipynb) as .shp files preview is not supported here!

# # Machine Learning Model

# ### Predicting the category of San Francisco crimes

# In[ ]:


# encode crime categories
le = preprocessing.LabelEncoder()
category = le.fit_transform(train['Category'])


# In[ ]:


# encode weekdays, districts and hours
district = pd.get_dummies(train['PdDistrict'])
days = pd.get_dummies(train['DayOfWeek'])
train['Dates'] = pd.to_datetime(train['Dates'], format = '%Y/%m/%d %H:%M:%S')
hour = train['Dates'].dt.hour
hour = pd.get_dummies(hour)


# In[ ]:


# pass encoded values to a new dataframe
enc_train = pd.concat([hour, days, district], axis = 1)
enc_train['Category'] = category


# In[ ]:


# add gps coordinates
enc_train['X'] = train['X']
enc_train['Y'] = train['Y']


# In[ ]:


# repeat data handling for test data by encoding weekdays, districts and hours
district = pd.get_dummies(test['PdDistrict'])
days = pd.get_dummies(test['DayOfWeek'])
test['Dates'] = pd.to_datetime(test['Dates'], format = '%Y/%m/%d %H:%M:%S')
hour = test['Dates'].dt.hour
hour = pd.get_dummies(hour)


# In[ ]:


# create a new dataframe for encoded test values
enc_test = pd.concat([hour, days, district], axis = 1)


# In[ ]:


# add gps coordinates
enc_test['X'] = test['X']
enc_test['Y'] = test['Y']


# Next step is to split up the enc_train into a training and validation set so that we there's a way to access the model performance without touching the test data.

# In[ ]:


training, validation = train_test_split(enc_train, train_size = 0.60)


# In[ ]:


features = ['Friday', 'Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN', 'X', 'Y']
# add the time
features.extend(x for x in range(0,24))


# In[ ]:


start = time.time()
model = BernoulliNB()
model.fit(training[features], training['Category'])
predicted = np.array(model.predict_proba(validation[features]))
end = time.time()
secs = (end - start)
loss = log_loss(validation['Category'], predicted)
print("Total seconds: {} and loss {}".format(secs, loss))


# In[ ]:


# now let's see what log_loss score we get if we apply LogisticRegression
start = time.time()
model = LogisticRegression(C = 0.01)
model.fit(training[features], training['Category'])
predicted = np.array(model.predict_proba(validation[features]))
end = time.time()
secs = (end - start)
loss = log_loss(validation['Category'], predicted)
print("Total seconds: {} and loss {}".format(secs, loss))


# log_loss or logarithmic loss is a classification metric based on probabilities where it quantifies the accuracy of a classifier by penalising false classifications, in other words minimising the log_loss means maximising the accuracy and lower log-loss value makes better predictions. Also BernoulliNB took only 2.6xx secs to run while LogisticRegression took much longer.

# In[ ]:


model = BernoulliNB()
model.fit(enc_train[features], enc_train['Category'])
predicted = model.predict_proba(enc_test[features])


# In[ ]:


# extract results
result = pd.DataFrame(predicted, columns = le.classes_)
result.to_csv('results.csv', index = True, index_label = 'Id')


# However Naive Bayes is a fairly simple model, it can give great results!
