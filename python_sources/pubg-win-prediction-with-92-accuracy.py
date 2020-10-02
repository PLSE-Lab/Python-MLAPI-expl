#!/usr/bin/env python
# coding: utf-8

# **PUBG** Win Prediction

# In[ ]:


from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
import matplotlib as mpl
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import numpy
import json
import sys
import csv
import os


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_train=pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')


# In[ ]:


df_train.shape


# In[ ]:


df_train.columns


# In[ ]:


#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[ ]:


#function for missing data
def missing_data(df_train):
    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return(missing_data.head(20))


# In[ ]:


missing_data(df_train)


# In[ ]:


df_train.winPlacePerc=df_train.winPlacePerc.fillna('0.4782')


# In[ ]:


df_train['matchType'].value_counts()


# In[ ]:


df_train=df_train.drop(['Id', 'groupId', 'matchId'],axis=1)


# In[ ]:


df_train.info()


# In[ ]:


df_train['winPlacePerc']=df_train['winPlacePerc'].astype("float")


# In[ ]:


encoded=pd.get_dummies(df_train)


# In[ ]:


encoded.head()


# In[ ]:


dependent_all=encoded['winPlacePerc']


# In[ ]:


independent_all=encoded.drop(['winPlacePerc'],axis=1)


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(independent_all,dependent_all,test_size=0.3,random_state=100)


# In[ ]:


linregr = LinearRegression()
linregr.fit(x_train, y_train)
pred_linreg = linregr.predict(x_train)


# In[ ]:



print("accuracy score for train using linearregression is",linregr.score(x_train,y_train))
print("accuracy score for test using linearregression is",linregr.score(x_test,y_test))


# In[ ]:


#random Forest
rfr = RandomForestRegressor()


# In[ ]:


rfr.fit(x_train,y_train)


# In[ ]:


predicted = rfr.predict(x_train)


# In[ ]:


print("accuracy score for train using randomforest",rfr.score(x_train,y_train))
print("accuracy score for test using randomforest",rfr.score(x_test,y_test))


# In[ ]:


#decisiontreeregressor
dtr = DecisionTreeRegressor()


# In[ ]:


dtr.fit(x_train,y_train)


# In[ ]:


predicted=dtr.predict(x_train)


# In[ ]:


print("accuracy score for train using Decision_tree",dtr.score(x_train,y_train))
print("accuracy score for test using Decision_tree",dtr.score(x_test,y_test))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




