#!/usr/bin/env python
# coding: utf-8

# # Let us the fun begin
# **To cut long story short**, the provided data is to small to siginifciantly distinguish beer's style by ibu or abv.

# ## import beers and breweries

# In[1]:


#import libraries
import pandas as pd
import numpy as np


# In[2]:


#import beers
beers = pd.read_csv('../input/beers.csv', index_col='name')


# In[3]:


#import breweries
breweries = pd.read_csv('../input/breweries.csv', index_col='name')


# # Can you predict the beer type from the characteristics provided in the dataset?

# ## Clearing the beer dataset
# To predict the beer type, I'm going to use abv and ibu.
# <br>**ABV** stands for **alcohol by volume** and describes ... the dose of fairytale in the bottle. Seriously.
# <br>**IBU** stands for **International Bittering Units** scale and describes the amount of iso-alpha acids.

# Drop NaNs

# In[4]:


sbeers = beers[
    ['abv','ibu','style']
]
sbeers = sbeers.dropna()


# Split the beers data into train and test

# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(sbeers[['ibu','abv']], sbeers[['style']], random_state=0)
from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)


# ## Using logistic regression

# In[6]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='liblinear', multi_class='ovr')
logreg.fit(X_train, np.ravel(y_train), )
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))


# ## Using random forests

# In[7]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
fvals = le.fit_transform(np.ravel(sbeers[['style']]))
fvals = np.unique(fvals)


# In[8]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1001)
rf.fit(X_train, le.transform(np.ravel(y_train)));


# In[9]:


predictions = rf.predict(X_test)


# Try to match prediction to the closes label

# In[10]:


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# In[11]:


for i in range (len(predictions)):
    predictions[i] = find_nearest(fvals, predictions[i])


# In[12]:


predictions = predictions.astype(int)
predictions = le.inverse_transform(predictions)


# In[13]:


foo = pd.DataFrame({'actual_values' : y_test['style'], 'predictions':predictions})
foo['comp'] = np.where(foo['actual_values']==foo['predictions'], 1, 0)
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format((len(foo[foo['comp']==1])/len(foo))))


# ## Building a linear scale

# The idea was to check if naive sorting beers by ibu or abv is enough to see cutoff levels, to build a kind of scale.

# In[14]:


tr_sc = pd.concat([X_train,y_train], axis=1)
tr_sc = tr_sc.sort_values(by='abv')


# As can be seen below, there is no a clear correlation between abv and a style. So, the simple scale is not available to clearly distinguish beer styles.

# In[15]:


tr_sc.tail(20)


# ## Using SVM

# In[16]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
#svclassifier = SVC(kernel='Gaussian')
svclassifier.fit(X_train, np.ravel(y_train));


# In[17]:


y_pred = svclassifier.predict(X_test)


# In[18]:


foo = pd.DataFrame({'actual_values' : y_test['style'], 'predictions':y_pred})
foo['comp'] = np.where(foo['actual_values']==foo['predictions'], 1, 0)
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format((len(foo[foo['comp']==1])/len(foo))))


# # What is the most popular beer in North Dakota?
# 
# 

# In[19]:


beers.head()


# In[20]:


breweries = pd.read_csv('../input/breweries.csv')
breweries = breweries.rename(columns ={'Unnamed: 0':'brewery_id'})


# In[21]:


cdata = beers.merge(breweries, on='brewery_id')
cdata['state'] = cdata['state'].str.strip()


# In[22]:


cdata[cdata['state']=="ND"]

