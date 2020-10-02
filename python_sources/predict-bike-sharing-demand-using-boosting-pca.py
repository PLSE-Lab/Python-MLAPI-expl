#!/usr/bin/env python
# coding: utf-8

# ##Predict Bike Sharing Demand using Boosting & PCA##

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import seaborn as sns


# In[ ]:


dataset = pd.read_csv('../input/train.csv')


# In[ ]:


dataset.head()


# In[ ]:


dataset.datetime = dataset.datetime.apply(pd.to_datetime)
dataset['year'] = dataset.datetime.apply(lambda x: x.year)
dataset['month'] = dataset.datetime.apply(lambda x: x.month)
dataset['day'] = dataset.datetime.apply(lambda x: x.day)
dataset['hour'] = dataset.datetime.apply(lambda x: x.hour)
dataset.drop('datetime', axis=1, inplace=True)
dataset.head()


# In[ ]:


from sklearn.model_selection import train_test_split
dataset.keys()


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()
scaler.fit(dataset[[u'season', u'holiday', u'workingday', u'weather', u'temp', u'atemp', u'humidity', u'windspeed', u'year', u'month', u'day', u'hour']])
scaled_dataset = scaler.transform(dataset[[u'season', u'holiday', u'workingday', u'weather', u'temp', u'atemp', u'humidity', u'windspeed', u'year', u'month', u'day', u'hour']])
scaled_dataset = pd.DataFrame(scaled_dataset, columns=[u'season', u'holiday', u'workingday', u'weather', u'temp', u'atemp', u'humidity', u'windspeed', u'year', u'month', u'day', u'hour'])
scaled_dataset.head()


# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca = PCA(n_components=2)
pca.fit(dataset)
pca_dataset = pca.transform(dataset)


# In[ ]:


#X = scaled_dataset[[u'season', u'holiday', u'workingday', u'weather', u'temp', u'atemp', u'humidity', u'windspeed', u'year', u'month', u'day', u'hour']]
X = pca_dataset #use either scaled_dataset or pca_dataset. using pca_dataset here as it gave better R-squared score for training dataset
y = dataset[u'count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


# In[ ]:


from sklearn.ensemble import AdaBoostRegressor


# In[ ]:


model = AdaBoostRegressor()


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


model.score(X_test, y_test)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


plt.scatter(y_test, y_pred)


# In[ ]:


test_dataset = pd.read_csv('../input/test.csv')
datetime = test_dataset['datetime']


# In[ ]:


test_dataset.datetime = test_dataset.datetime.apply(pd.to_datetime)
test_dataset['year'] = test_dataset.datetime.apply(lambda x: x.year)
test_dataset['month'] = test_dataset.datetime.apply(lambda x: x.month)
test_dataset['day'] = test_dataset.datetime.apply(lambda x: x.day)
test_dataset['hour'] = test_dataset.datetime.apply(lambda x: x.hour)
test_dataset.drop('datetime', axis=1, inplace=True)
test_dataset.head()


# In[ ]:


test_dataset.keys()


# In[ ]:


scaler = StandardScaler()
scaler.fit(test_dataset[[u'season', u'holiday', u'workingday', u'weather', u'temp', u'atemp', u'humidity', u'windspeed', u'year', u'month', u'day', u'hour']])
scaled_test_dataset = scaler.transform(test_dataset[[u'season', u'holiday', u'workingday', u'weather', u'temp', u'atemp', u'humidity', u'windspeed', u'year', u'month', u'day', u'hour']])
scaled_test_dataset = pd.DataFrame(scaled_test_dataset, columns=[u'season', u'holiday', u'workingday', u'weather', u'temp', u'atemp', u'humidity', u'windspeed', u'year', u'month', u'day', u'hour'])
scaled_test_dataset.head()


# In[ ]:


pca = PCA(n_components=2)
pca.fit(test_dataset)
pca_test_dataset = pca.transform(test_dataset)


# In[ ]:


#test_pred = model.predict(scaled_test_dataset)
test_pred = model.predict(pca_test_dataset) #use either scaled_dataset or pca_dataset. using pca_dataset here as it gave better R-squared score for training dataset


# In[ ]:


submission = pd.DataFrame({'datetime': datetime, 'count': test_pred})


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('actualSubmission.csv', index=False)


# In[ ]:




