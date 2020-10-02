#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))


# In[ ]:


#Importing the dataset
dataset = pd.read_csv('../input/Social_Network_Ads.csv')
X= dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values
dataset.head()


# In[ ]:


#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# First of all let's convert training data into LightGBM dataset format ( this is mandatory for LightGBM to work)

# In[ ]:


import lightgbm as lgb

d_train = lgb.Dataset(x_train, label= y_train)


# Now let's create dictionary for parameters values

# In[ ]:


params = {}
params['learning_rate']= 0.003
params['boosting_type']='gbdt'
params['objective']='binary'
params['metric']='binary_logloss'
params['sub_feature']=0.5
params['num_leaves']= 10
params['min_data']=50
params['max_depth']=10


# Now train model with 100 iterations

# In[ ]:


clf= lgb.train(params, d_train, 100)


# final predict the output. Output will be a list of probabilities.

# In[ ]:


y_pred = clf.predict(x_test)

#convert into binary values

for i in range(0,100):
    if (y_pred[i] >= 0.5):
        y_pred[i] = 1
    else:
        y_pred[i] =0
len(y_pred)        
            


# we can check the results either using confusion matrix or directly calculating accuracy

# In[ ]:



#Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_pred, y_test)
accuracy


# In[ ]:


#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# Complete tutorial can be found [here](https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc).
# Thanks to Pushkar Mandot for the  tutorial.
