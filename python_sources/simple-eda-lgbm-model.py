#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import sklearn.linear_model
import sklearn.model_selection
import pandas as pd


# In[ ]:


train =  pd.read_csv('../input/hse-aml-2020/books_train.csv')


# In[ ]:


train_x = train.drop('average_rating', axis=1)
train_y = train.average_rating


# In[ ]:


train_x


# In[ ]:


train_x['publication_date'] = pd.to_datetime(train_x['publication_date'], format='%m/%d/%Y', errors='coerce')
train_x['year'] = pd.to_numeric(train_x['publication_date'].dt.year, downcast='integer')
train_x['month'] = pd.to_numeric(train_x['publication_date'].dt.month, downcast='integer')
train_x['day'] = pd.to_numeric(train_x['publication_date'].dt.day, downcast='integer')


# In[ ]:


train_x['language_code'] = train_x['language_code'].replace(to_replace=train_x.language_code.unique(), value=np.arange(0, len(train_x.language_code.unique()), 1))


# In[ ]:


train_x


# In[ ]:


train_x['authors'] = train_x['authors'].replace(to_replace=train_x.authors.unique(), value=np.arange(0, len(train_x.authors.unique()), 1))
train_x['publisher'] = train_x['publisher'].replace(to_replace=train_x.publisher.unique(), value=np.arange(0, len(train_x.publisher.unique()), 1))


# In[ ]:


del train_x['title']
del train_x['authors']
del train_x['publisher']
del train_x['publication_date']


# In[ ]:


del train_x['isbn']


# In[ ]:


from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

light = LGBMRegressor()
light.fit(train_x, train_y)
valid_pred_merged = light.predict(train_x)
print('MSE: {}'.format(mean_squared_error(train_y, valid_pred_merged)))


# In[ ]:


test = pd.read_csv('../input/hse-aml-2020/books_test.csv')


# # Lets make test dataset looks like train dataset

# In[ ]:


test['publication_date'] = pd.to_datetime(test['publication_date'], format='%m/%d/%Y', errors='coerce')
test['year'] = pd.to_numeric(test['publication_date'].dt.year, downcast='integer')
test['month'] = pd.to_numeric(test['publication_date'].dt.month, downcast='integer')
test['day'] = pd.to_numeric(test['publication_date'].dt.day, downcast='integer')


# In[ ]:


test['language_code'] = test['language_code'].replace(to_replace=test.language_code.unique(), value=np.arange(0, len(test.language_code.unique()), 1))


# In[ ]:


test['authors'] = test['authors'].replace(to_replace=test.authors.unique(), value=np.arange(0, len(test.authors.unique()), 1))
test['publisher'] = test['publisher'].replace(to_replace=test.publisher.unique(), value=np.arange(0, len(test.publisher.unique()), 1))


# In[ ]:


del test['title']
del test['authors']
del test['publisher']
del test['publication_date']


# In[ ]:


del test['isbn']


# In[ ]:


valid_pred_merged = light.predict(test)
#print('MSE: {}'.format(mean_squared_error(test, valid_pred_merged)))


# In[ ]:


pd.Series(valid_pred_merged)


# In[ ]:


test['bookID']


# In[ ]:


df = pd.DataFrame(np.nan, index=[0,1,2,3], columns=['A'])
df['bookID'] = test['bookID']
df['average_rating'] = pd.Series(valid_pred_merged)#bookID,average_rating

d = {'bookID': test['bookID'], 'average_rating': pd.Series(valid_pred_merged)}
df = pd.DataFrame(data=d)


# In[ ]:


df


# In[ ]:


df.to_csv('file_name.csv', index=False)

