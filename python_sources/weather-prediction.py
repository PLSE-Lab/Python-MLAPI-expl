#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
import xgboost as xgb
from sklearn.ensemble import BaggingRegressor
import numpy as np 
import pandas as pd 
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score


# In[ ]:


data = pd.read_csv('/kaggle/input/us-weather-events/US_WeatherEvents_2016-2019.csv')


# In[ ]:


data.isna().sum()


# In[ ]:


data = data.fillna(data.median())


# In[ ]:


data.head()


# In[ ]:


table_1 = data.groupby(['City','Type']).count()


# In[ ]:


table_1 = table_1.reset_index()


# In[ ]:


table_1


# Here I collect the number of types according to cities and then I will merge them with main dataframe.

# In[ ]:


list_of_dic = []
label = table_1.City[0]
list_val = {}
for index,row in table_1.iterrows():   
    if row['City'] != label:
        label = row['City']
        list_of_dic.append(list_val)
        list_val = {}
        list_val['City'] = row['City']
        list_val[row['Type']] = row['EventId']
       
    else:
        list_val['City'] = row['City']
        list_val[row['Type']] = row['EventId']
        
    
    
    


# In[ ]:


df = pd.DataFrame()
for dic in list_of_dic:
    df = df.append(dic, ignore_index=True)
    


# In[ ]:


df = df.fillna(0)


# In[ ]:


data = data.merge(df,on = 'City')


# In[ ]:


data


# In[ ]:


data['StartTime(UTC)'] = pd.to_datetime(data['StartTime(UTC)'])
data['EndTime(UTC)'] = pd.to_datetime(data['EndTime(UTC)'])

data['Start_year'] = data['StartTime(UTC)'].dt.year
data['Start_month'] = data['StartTime(UTC)'].dt.month
data['Start_week'] = data['StartTime(UTC)'].dt.week
data['Start_weekday'] = data['StartTime(UTC)'].dt.weekday
data['Start_day'] = data['StartTime(UTC)'].dt.day

data['end_year'] = data['EndTime(UTC)'].dt.year
data['end_month'] = data['EndTime(UTC)'].dt.month
data['end_week'] = data['EndTime(UTC)'].dt.week
data['end_weekday'] = data['EndTime(UTC)'].dt.weekday
data['end_day'] = data['EndTime(UTC)'].dt.day


# In[ ]:


X = data.drop(['Type','StartTime(UTC)','EndTime(UTC)'],axis = 1).head(5000)
y = data.Type.head(5000)


# In[ ]:


le = preprocessing.LabelEncoder()
for name in X.columns:
    if X[name].dtypes == "O":
        print(name)
        X[name] = X[name].astype(str)
        le.fit(X[name])
        X[name] = le.transform(X[name])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


clf = RandomForestClassifier(n_estimators = 400,min_samples_split = 2,min_samples_leaf = 1,max_features= 'sqrt',max_depth =None,bootstrap= False)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)


# In[ ]:


X


# In[ ]:


import seaborn as sns
ax = sns.barplot(x=clf.feature_importances_, y=X.columns)


# In[ ]:


accuracy_score(predictions,y_test)


# In[ ]:




