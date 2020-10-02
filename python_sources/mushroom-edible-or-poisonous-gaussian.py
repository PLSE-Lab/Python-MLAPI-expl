#!/usr/bin/env python
# coding: utf-8

# In[114]:


# Required Python Machine learning Packages
import pandas as pd
import numpy as np
# For preprocessing the data
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
# To split the dataset into train and test datasets
from sklearn.model_selection import train_test_split
# To model the Gaussian Navie Bayes classifier
from sklearn.naive_bayes import GaussianNB
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score
import math


# In[115]:


path="../input/mushrooms.csv"


# In[116]:


headers=['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color',
         'stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring',
         'veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']


# In[117]:


data=pd.read_csv(path)


# In[118]:


data.isnull().values.sum()


# In[119]:


data.columns=headers


# In[120]:


data_b=data


# In[121]:


data.head()


# In[122]:


for h in headers:
    le = preprocessing.LabelEncoder()
    encoded_h = le.fit_transform(data[h])
    data_b[h+"_cat"] = encoded_h


# In[123]:


data_b.head()


# In[124]:


data_b = data_b.drop(headers, axis = 1)


# In[125]:


data_b.head()


# In[126]:


# data_b=data.reindex_axis(headers,axis=1)
data_b=data_b.reset_index()


# In[127]:


data_b.head()


# In[133]:


for h in headers:
    if(h is "class"):
        print("skipping..")
    else:
        mean,std=data_b[h+"_cat"].mean(),data_b[h+"_cat"].std()
        if(mean != 0.0 and std != 0.0):
            std_v=(data_b[h+"_cat"] - mean)/std
            data_b.loc[:, h+"_cat"] =  std_v
        else:
            data_b.loc[:, h+"_cat"] =  0.0


# In[134]:


data_b.head(5)


# In[145]:


features = data_b.values[:,2:23]
target = data_b.values[:,1]
X_train, X_test, y_train, y_test = train_test_split(features,target, test_size = 0.33, random_state = 10)


# In[146]:


data_b.isnull().sum()


# In[147]:


clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[148]:


accuracy_score(y_test, y_pred, normalize = True)


# In[ ]:




