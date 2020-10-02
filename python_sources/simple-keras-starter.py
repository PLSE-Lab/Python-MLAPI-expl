#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd


# # Loading the data

# Loading the train data and creating simple features

# In[ ]:


train_data = pd.read_csv("../input/train.csv", parse_dates=True, index_col =0)
y_train = train_data.iloc[:,-1].values
train_data.drop('sales', 1, inplace=True)
train_data['y'] = train_data.index.year-train_data.index.year.min()
train_data['m'] = train_data.index.month
train_data['d'] = train_data.index.day
train_data['dow'] = train_data.index.dayofweek


# Splitting the data if cross validation is needed, but for the submission it is not needed.

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(train_data, y_train, test_size=.1, random_state=0, shuffle = True)


# Determining the categorical variables. Only year is not categorical, as the value of years are numerically comparable.

# In[ ]:


cat_vars = list(train_data.columns)
cat_vars.remove('y')
cont_vars = ['y']


# Structuring data for training and validation. You will see in the Keras model why we had to form the data like this( A list of numpy arrays).

# In[ ]:


X_train = []
X_val = []
X_train.append(x_train[cont_vars].astype('float32').values)
X_val.append(x_val[cont_vars].astype('float32').values)
for cat in cat_vars:
    X_train.append(x_train[cat].values)
    X_val.append(x_val[cat].values)


# Determining the embedding size for each category. The formula has been working good in practice.

# In[ ]:


cat_sizes = {}
cat_embsizes = {}
for cat in cat_vars:
    cat_sizes[cat] = train_data[cat].nunique()
    cat_embsizes[cat] = min(50, cat_sizes[cat]//2+1)


# In[ ]:


cat_embsizes


# Loading the test data.

# In[ ]:


test_data = pd.read_csv("../input/test.csv", parse_dates=True, index_col =1)
test_data['y'] = test_data.index.year-train_data.index.year.min()
test_data['m'] = test_data.index.month
test_data['d'] = test_data.index.day
test_data['dow'] = test_data.index.dayofweek
# test_data['special_store'] = test_data['store'].isin([5,6,7])*1


# In[ ]:


X_test = []
X_test.append(test_data[cont_vars].astype('float32').values)
for cat in cat_vars:
    X_test.append(test_data[cat].values)


# # Keras Model

# In[ ]:


from keras.layers import Dense, Dropout, Embedding, Input, Reshape, Concatenate
from keras.models import Model


# Each category has to have its own embedding matrix so they should be individually added as inputs. A very simple model of input-> embedding-> dense-> dense-> output is used

# In[ ]:


import keras.backend as K

def custom_smape(x, x_):
    return K.mean(2*K.abs(x-x_)/(K.abs(x)+K.abs(x_)))


# In[ ]:


y = Input((len(cont_vars),), name='cont_vars')
ins = [y]
concat = [y]
for cat in cat_vars:
    x = Input((1,), name=cat)
    ins.append(x)
    x = Embedding(cat_sizes[cat]+1, cat_embsizes[cat], input_length=1)(x)
    x = Reshape((cat_embsizes[cat],))(x)
    concat.append(x)
y = Concatenate()(concat)
y = Dense(100, activation= 'relu')(y)
# y = Dense(100, activation= 'relu')(y)
y = Dense(1)(y)
model = Model(ins, y)
model.summary()
model.compile('adadelta', custom_smape)


# Only 2 epochs are enough to train the network.

# In[ ]:


model.fit(X_train, y_train, 64, 2, validation_data=[X_val, y_val])


# Submitting the test results.

# In[ ]:


test_preds = model.predict(X_test)
sample_data = pd.read_csv("../input/sample_submission.csv", index_col=0)
sample_data['sales'] = test_preds
sample_data.to_csv('preds.csv')

