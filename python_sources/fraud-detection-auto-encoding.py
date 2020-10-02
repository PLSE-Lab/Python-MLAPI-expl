#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### Load required libraries

# In[ ]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from keras.layers import Input, Dense, Dropout
from keras.models import Model

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Reading the data

# In[ ]:


data = pd.read_csv("../input/Fraud_data_amtstd.csv")


# ### Understand the data

# #### Display No. of records and attributes 

# In[ ]:


data.shape


#  Observation: 
#  
#         Data has 30 attributes and 1 lakh records

# #### Look at first 5 records

# In[ ]:


data.head()


# #### Display column names

# In[ ]:


data.columns


# #### Data type of each attribute

# In[ ]:


data.dtypes


# #### Data distribution w.r.t target attributes

# In[ ]:


print(pd.value_counts(data['Class']))

print(pd.value_counts(data['Class'])/data['Class'].shape[0])


# #### Bar plot

# In[ ]:


# Drawing a barplot
pd.value_counts(data['Class']).plot(kind = 'bar', rot=0)

# Giving titles and labels to the plot
plt.title("Transaction class distribution")
plt.xticks(range(2), ["Normal", "Fraud"])
plt.xlabel("Class")
plt.ylabel("Frequency");


#     O = Normal
# 
#     1 = Fraud

# #### Extract numpy array from the DataFrame

# In[ ]:


data = data.values


# ### Train test split
# 
#     Splitting the data into train and test, such that train data has only non-fraud records and test data has both. 

# In[ ]:


data_nf = data[data[:,-1] == 0]
test_f  = data[data[:,-1] == 1]

train_nf, test_nf = train_test_split(data_nf, test_size=0.2, random_state=123)


# In[ ]:


print(data.shape)
print(train_nf.shape)
print(test_nf.shape)
print(test_f.shape)


# #### Look at the distribution w.r.t target attribute

# In[ ]:


print(np.unique(data[:,-1], return_counts=True))
print(np.unique(train_nf[:,-1], return_counts=True))
print(np.unique(test_nf[:,-1], return_counts=True))
print(np.unique(test_f[:,-1], return_counts=True))


# #### Only extract independent features

# In[ ]:


X_train_nf = train_nf[:,:-1]

X_test_nf = test_nf[:,:-1]

X_test_f = test_f[:,:-1]


# ### Build Autoencoder

# In[ ]:


input_dim = X_train_nf.shape[1]
#encoding_dim = 15
encoding_dim = 150


# In[ ]:


# Input placeholder
input_att = Input(shape=(input_dim,))

input_dropout = Dropout(0.2)(input_att)
 
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_dropout)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(input_dim, activation='linear')(encoded)


# In[ ]:


autoencoder = Model(input_att, decoded)


# #### Compile the model

# In[ ]:


autoencoder.compile(loss='mean_squared_error', optimizer='adam')


# #### Fit the model

# In[ ]:


get_ipython().run_line_magic('time', 'autoencoder.fit(X_train_nf, X_train_nf, epochs=50, shuffle=True, validation_split=0.2, verbose=1)')


# #### Evaluate the loss on non-fraud train data

# In[ ]:


autoencoder.evaluate(X_train_nf, X_train_nf)


# #### Evaluate the loss on non-fraud test data

# In[ ]:


autoencoder.evaluate(X_test_nf, X_test_nf)


# #### Evaluate the loss on fraud test data

# In[ ]:


autoencoder.evaluate(X_test_f, X_test_f)


# #### Function to calculate mse for each record

# In[ ]:


def mse_for_each_record(act, pred):
    error = act - pred
    squared_error = np.square(error)
    mean_squared_error = np.mean(squared_error, axis=1)
    return mean_squared_error


# #### Making predictions on the non-fraud train data

# In[ ]:


pred_train_nf = autoencoder.predict(X_train_nf)

mse_train_nf = mse_for_each_record(X_train_nf, pred_train_nf)


# #### Making predictions on the non-fraud test data

# In[ ]:


pred_test_nf = autoencoder.predict(X_test_nf)

mse_test_nf = mse_for_each_record(X_test_nf, pred_test_nf)


# #### Making predictions on the fraud test data

# In[ ]:


pred_test_f = autoencoder.predict(X_test_f)

mse_test_f = mse_for_each_record(X_test_f, pred_test_f)


# ### Explore and identify right cut-off 

# #### mse box plots of non-fraud train, non-fraud test and fraud test data

# In[ ]:


plt.subplot(1, 3, 1)
plt.boxplot(mse_train_nf)

plt.subplot(1, 3, 2)
plt.boxplot(mse_test_nf)

plt.subplot(1, 3, 3)
plt.boxplot(mse_test_f)


# #### Summary statistics on mse of non-fraud train, non-fraud test and fraud test data 

# In[ ]:


print("-------mse_train_nf-------")
print(pd.Series(mse_train_nf).describe())
print("\n-------mse_test_NF-------")
print(pd.Series(mse_test_nf).describe())
print("\n-------mse_test_f-------")
print(pd.Series(mse_test_f).describe())


# #### Decide cut-off

# In[ ]:


cut_off = np.round(np.percentile(mse_train_nf,99),2)

print("Cut-off = {}".format(cut_off))


# #### % of correctly predicted non-fraud train, non-fraud test and fraud test records

# In[ ]:


print("Non-fraud train records = {}%".format(np.round(np.sum(mse_train_nf <= cut_off)/train_nf.shape[0],2)*100))
print("Non-fraud test records  = {}%".format(np.round(np.sum(mse_test_nf <= cut_off)/test_nf.shape[0],2)*100))
print("Fraud test records      = {}%".format(np.round(np.sum(mse_test_f > cut_off)/test_f.shape[0],2)*100))


# In[ ]:




