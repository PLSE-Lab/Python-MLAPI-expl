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
print(os.listdir("../input/black friday/black friday"))
import xgboost
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor 
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, Activation
# Any results you write to the current directory are saved as output.


# In[ ]:


train=pd.read_csv("../input/black friday/black friday/train_modified.csv")
Test=pd.read_csv("../input/black friday/black friday/test_modified.csv")
purchase=pd.read_csv("../input/black friday/black friday/train.csv")


# In[ ]:


train.drop('0',inplace=True,axis=1)
train.dropna(how='all',axis=0,inplace=True)
Y=purchase['Purchase']


# In[ ]:


train.shape


# In[ ]:


Test.shape


# In[ ]:





# In[ ]:


train.shape


# In[ ]:


train.info()


# In[ ]:


Test.info()


# In[ ]:





# In[ ]:


train.shape


# In[ ]:


X=train


# In[ ]:


X['train']=1
Test['train']=0


# In[ ]:


combined=pd.concat([X,Test])


# In[ ]:


combined.info()


# In[ ]:


le=LabelEncoder()
combined['User_ID']=le.fit_transform(combined['User_ID'])


# In[ ]:


X=combined[combined['train']==1]
Test=combined[combined['train']==0]
X.drop('train',inplace=True,axis=1)
Test.drop('train',inplace=True,axis=1)


# In[ ]:


Test.shape


# In[ ]:


Test.info()


# In[ ]:


X=X.astype(np.float32)
Test=Test.astype(np.float32)


# In[ ]:


#a=X.loc[:,['Product_ID_mean','Product_ID_max','User_ID_min','Product_ID_min','User_ID','Product_ID','Age']]


# In[ ]:


#a.head()


# In[ ]:



#x_train, x_test, y_train, y_test = train_test_split(a, Y, test_size=0.1, random_state=42)


# In[ ]:


#x_train.shape


# In[ ]:


import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
h2o.init(max_mem_size = 2)
h2o.remove_all()
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator, H2ODeepLearningEstimator
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator, H2ODeepLearningEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator


# In[ ]:


X['y']=Y
X.head()


# In[ ]:


a=list(X)


# In[ ]:


predictors=a[0:33]
response=a[33]


# In[ ]:


df = h2o.H2OFrame(X)


# In[ ]:


train,test,valid = df.split_frame(ratios=list([.7, .15]))


# In[ ]:


dl_model_1 = H2ODeepLearningEstimator(epochs=40,
                                      hidden=[60],
                                      activation="Rectifier",
                                      adaptive_rate =False )


# In[ ]:


dl_model_1.train(x = predictors, y = response, training_frame = train, validation_frame = valid)
y_pred=dl_model_1.predict(test)
abc= h2o.as_list(y_pred, use_pandas=True) 
xyz=h2o.as_list(test['y'], use_pandas=True) 
print(sqrt(mean_squared_error(xyz, abc)))


# In[ ]:


dl_model_2 = H2ODeepLearningEstimator(epochs=120,
                                      hidden=[60],
                                      activation="Rectifier",
                                      adaptive_rate =False )


# In[ ]:


dl_model_2.train(x = predictors, y = response, training_frame = train, validation_frame = valid)
y_pred=dl_model_2.predict(test)
abc= h2o.as_list(y_pred, use_pandas=True) 
xyz=h2o.as_list(test['y'], use_pandas=True) 
print(sqrt(mean_squared_error(xyz, abc)))


# In[ ]:


dl_model_3 = H2ODeepLearningEstimator(epochs=60,
                                      hidden=[6],
                                      activation="Rectifier",
                                      adaptive_rate =False )


# In[ ]:


dl_model_3.train(x = predictors, y = response, training_frame = train, validation_frame = valid)
y_pred=dl_model_3.predict(test)
abc= h2o.as_list(y_pred, use_pandas=True) 
xyz=h2o.as_list(test['y'], use_pandas=True) 
print(sqrt(mean_squared_error(xyz, abc)))


# In[ ]:





# In[ ]:





# In[ ]:


gbm_1 = H2OGradientBoostingEstimator(max_depth=3,
           distribution = "gaussian",
           ntrees =500,
           learn_rate = 0.05,
           nbins_cats = 5891)


# In[ ]:


gbm_1.train(x = predictors, y = response, training_frame = train, validation_frame = valid)
y_pred=gbm_1.predict(test)
abc= h2o.as_list(y_pred, use_pandas=True) 
xyz=h2o.as_list(test['y'], use_pandas=True)
print(sqrt(mean_squared_error(xyz, abc)))


# In[ ]:





# In[ ]:


gbm_2 = H2OGradientBoostingEstimator(max_depth=3,
           distribution = "gaussian",
           ntrees =430,
           learn_rate = 0.04,
           nbins_cats = 5891)


# In[ ]:


gbm_2.train(x = predictors, y = response, training_frame = train, validation_frame = valid)
y_pred=gbm_2.predict(test)
abc= h2o.as_list(y_pred, use_pandas=True) 
xyz=h2o.as_list(test['y'], use_pandas=True)
print(sqrt(mean_squared_error(xyz, abc)))
           


# In[ ]:


abc=pd.DataFrame()
a=h2o.as_list(dl_model_1.predict(test), use_pandas=True) 

b=h2o.as_list(dl_model_2.predict(test), use_pandas=True) 
c=h2o.as_list(dl_model_3.predict(test), use_pandas=True) 
d=h2o.as_list(gbm_1.predict(test), use_pandas=True) 
e=h2o.as_list(gbm_2.predict(test), use_pandas=True) 


# In[ ]:


y=test['y']


# In[ ]:





# In[ ]:


abc['a']=a['predict']
abc['b']=b['predict']
abc['c']=c['predict']
abc['d']=d['predict']
abc['e']=e['predict']


# In[ ]:


xyz=h2o.as_list(test['y'], use_pandas=True)


# In[ ]:


abc['y']=xyz['y']


# In[ ]:


type(f)


# In[ ]:


abc.head()


# In[ ]:


abc.to_csv('dl_3_gbm_2.csv',index=False)


# In[ ]:


#pred_1=gbm.predict(test, num_iteration=gbm.best_iteration)


# In[ ]:


'''ub1=pd.DataFrame()
test=pd.read_csv("../input/black friday/black friday/test.csv")
sub1['User_ID']=test['User_ID']
sub1['Product_ID']=test['Product_ID']
sub1['Purchase']=dnn
sub1.to_csv('dnn.csv',index=False)'''


# In[ ]:


sub1.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




