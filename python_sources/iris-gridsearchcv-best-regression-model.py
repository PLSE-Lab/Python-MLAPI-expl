#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Measuring the score of Best Regression model by using the GridSearchCV
# 1. Linear Regression model
# 2. Decision Tree Regressor
# 3. Random Forest Regressor
# 4. XGBRegressor
# 5. KNeighborsRegressor
# 

# In[ ]:


data=pd.read_csv('/kaggle/input/iris/Iris.csv')
data.head()


# In[ ]:


data.columns


# In[ ]:


dummies = pd.get_dummies(data.Species)
dummies.head(3)


# In[ ]:


df11 = pd.concat([data,dummies],axis='columns')
df11.head()


# In[ ]:


df12 = df11.drop('Species',axis='columns')
df12.head(2)


# In[ ]:


X = df12.drop(['PetalWidthCm'],axis='columns')
X.head(3)


# In[ ]:


y = df12.PetalWidthCm
y.head(3)


# In[ ]:


len(y)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[ ]:


#Linear Regression model
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
model = LinearRegression()
# fit the model
model.fit(X, y)

# evaluate the model
yhat = model.predict(X_test)
model.score(X_test,y_test)


# In[ ]:


#Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
# define the model
model = DecisionTreeRegressor()
# fit the model
model.fit(X, y)
# evaluate the model
yhat = model.predict(X_test)
model.score(X_test,y_test)


# In[ ]:


# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
# define the model
model = RandomForestRegressor()
# fit the model
model.fit(X, y)
# evaluate the model
yhat = model.predict(X_test)
model.score(X_test,y_test)


# In[ ]:


#XGBRegressor
from xgboost import XGBRegressor
# define the model
model = XGBRegressor()
# fit the model
model.fit(X, y)
model.score(X_test,y_test)


# In[ ]:


#KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
# define the model
model = KNeighborsRegressor()
# fit the model
model.fit(X, y)
# evaluate the model
yhat = model.predict(X_test)
model.score(X_test,y_test)


# In[ ]:


from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        
        'DecisionTreeRegressor' : {
            'model': DecisionTreeRegressor(),
            'params': {
                
                'max_features': ['auto']
            }
        },
        
        'RandomForestRegressor' : {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': [100, 200, 300, 1000],
                'max_features': ['auto']              
                              
            }
        },
        
        'XGBRegressor' : {
            'model': XGBRegressor(),
            'params': {
                'booster': ['gbtree']             
                              
            }
        },
        
        'KNeighborsRegressor' : {
            'model': KNeighborsRegressor(),
            'params': {
                
                'algorithm' : ['kd_tree']              
                              
            }
        },
        
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)


# In[ ]:




