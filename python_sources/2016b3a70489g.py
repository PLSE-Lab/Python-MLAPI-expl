#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
get_ipython().system('{sys.executable} -m pip install --ignore-installed imblearn')


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df_train = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")
df_train.head()


# In[ ]:


df_submission = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/sample_submission.csv")
df_submission


# In[ ]:


df_train.info()


# In[ ]:


print(df_train.isnull().sum())


# In[ ]:


df_train.fillna(value=df_train.mean(),inplace=True)


# In[ ]:


print(df_train.isnull().sum())


# In[ ]:


df_train.isnull().any().any()


# In[ ]:


df_train.columns


# In[ ]:





# In[ ]:





# In[ ]:


numerical_features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5',
       'feature6', 'feature7', 'feature8', 'feature9', 'feature10',
       'feature11']
categorical_features = ['type']
X = df_train[numerical_features+categorical_features]
y = df_train['rating']


# In[ ]:


type_val = {'old':0,'new':1}
X['type'] = X['type'].map(type_val)
X.head()


# In[ ]:


y.value_counts()


# In[ ]:


y.value_counts()


# In[ ]:


df_train_new = df_train


# In[ ]:





# In[ ]:


df_train['rating'].value_counts()


# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.33,random_state=42,stratify=y)


# In[ ]:


from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_val[numerical_features] = scaler.transform(X_val[numerical_features])  

X_train[numerical_features].head()


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
#reg =  RandomForestClassifier(class_weight='balanced',n_estimators=400,max_depth=15).fit(X_train, y_train)
#clf = BalancedRandomForestClassifier(random_state=0,max_depth=5,n_estimators=500).fit(X_train, y_train)


# In[ ]:


#This is the model that got my highest score of 0.63810 on the private leaderboard
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
reg_ada = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=25,max_features=5),random_state=42).fit(X_train,y_train)


# In[ ]:


#This is the model that got my highest score of 0.64609 on the public leaderboard
from sklearn.ensemble import ExtraTreesRegressor
ET_reg = ExtraTreesRegressor(n_estimators=200,max_depth=25,random_state=52,max_features=4).fit(X_train, y_train)


# In[ ]:


y_pred = reg_ada.predict(X_val)


# In[ ]:


'''import math as m
def floattoint(x):
    for i in range(len(x)):
        if(x[i]<3):
            x[i] = m.ceil(x[i])
        else:
            x[i] = m.floor(x[i])
    return x'''


# In[ ]:


print(y_pred)


# In[ ]:


#y_pred = floattoint(y_pred)
#y_pred = y_pred.astype('int64')
#y_pred


# In[ ]:


y_pred = np.array(y_pred)
y_pred = y_pred.round()
y_pred = [int(i) for i in y_pred]


# In[ ]:


from sklearn.metrics import mean_squared_error

from math import sqrt

rmse = sqrt(mean_squared_error(y_pred, y_val))

print(rmse)


# In[ ]:


reg_full = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=25,max_features=5),random_state=42).fit(X, y)


# In[ ]:


df_test = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")
df_test.head()


# In[ ]:


df_test.info()


# In[ ]:


df_test.fillna(value=df_test.mean(),inplace=True)


# In[ ]:


df_test.isnull().any().any()


# In[ ]:


numerical_features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5',
       'feature6', 'feature7', 'feature8', 'feature9', 'feature10',
       'feature11']
categorical_features = ['type']
X_test = df_test[numerical_features+categorical_features]


# In[ ]:


type_val = {'old':0,'new':1}
X_test['type'] = X_test['type'].map(type_val)
X_test.head()


# In[ ]:


y_test = reg_full.predict(X_test)


# In[ ]:





# In[ ]:


y_test = np.array(y_test)
y_test = y_test.round()
y_test = [int(i) for i in y_test]


# In[ ]:





# In[ ]:


df_submission.columns


# In[ ]:


df_new = pd.DataFrame(columns = ['id', 'rating'])


# In[ ]:


df_new.head()


# In[ ]:


df_new["id"] = df_test["id"]
df_new["rating"] = y_test


# In[ ]:


df_new.to_csv("sub8.csv",index = False)


# In[ ]:




