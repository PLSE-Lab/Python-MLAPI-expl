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


# **Importing the required librarries**

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# **Importing the dataset**

# In[ ]:


get_ipython().system(' ls ../input/pima-indians-diabetes-database')


# In[ ]:


data = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')


# In[ ]:


data.head()


# **Basic EDA**

# In[ ]:


## gives information about the data types,columns, null value counts, memory usage etc
## function reference : https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.info.html
data.info(verbose=True)


# In[ ]:


print(data.describe())
print(data.shape)


# In[ ]:


# to check if any null value is present 
data.isnull().sum()


# In[ ]:


# checking corelation 
corr= data.corr()
plt.figure(figsize= (20,20))
sns.heatmap(corr,annot=True,cmap='winter_r')


# In[ ]:


data.corr()


# 
# outcome is already numerically classified. if Outcome was categrocial value(True/False)
# then we will map corresponding numerical value to Outcome.
# 
# Outcome = {True: 1, False: 0}
# 
# data['Outcome'] = data['Outcome'].map(diabetes_map)

# In[ ]:


sns.countplot(data.Outcome)
plt.xlabel('Outcome')
plt.ylabel('number of patient')
plt.show()


# In[ ]:


Outcome_true= len(data.loc[data['Outcome']==0])
Outcome_false= len(data.loc[data['Outcome']==1])
print(Outcome_true,Outcome_false)


# In[ ]:



## Train Test Split

from sklearn.model_selection import train_test_split
X =data.iloc[:,:-1].values
y= data.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=10)


# In[ ]:


print(X.shape,y.shape)


# many data conatins zeo values , which is not possible hence replancing zeo values with mean values of feature.
# 

# In[ ]:


print('total number of data {}'.format(data.shape[0]))
print('total number of prenancies {}'.format(len(data.loc[data['BMI']==0])))
print('total number of Glucose {}'.format(len(data.loc[data['Glucose']==0])))
print('total number of BloodPressure {}'.format(len(data.loc[data['BloodPressure']==0])))
print('total number of SkinThickness {}'.format(len(data.loc[data['SkinThickness']==0])))
print('total number of Insulin {}'.format(len(data.loc[data['Insulin']==0])))
print('total number of BMI {}'.format(len(data.loc[data['BMI']==0])))
print('total number of DiabetesPedigreeFunction {}'.format(len(data.loc[data['DiabetesPedigreeFunction']==0])))
print('total number of Age {}'.format(len(data.loc[data['Age']==0])))


# In[ ]:


from sklearn.impute import SimpleImputer

fill_values = SimpleImputer(missing_values=0, strategy="mean",verbose=0)

X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)


# In[ ]:


## Apply Algorithm

from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=10)

random_forest_model.fit(X_train, y_train)


# In[ ]:


predict_train_data = random_forest_model.predict(X_test)

from sklearn import metrics

print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))


# ** Hyper Parameter Optimization**
# 

# In[ ]:


params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}


# In[ ]:


## Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
import xgboost


# In[ ]:


classifier=xgboost.XGBClassifier()


# In[ ]:


random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)


# In[ ]:


random_search.fit(X_train,y_train)


# In[ ]:


random_search.best_score_


# In[ ]:


random_search.best_estimator_


# In[ ]:


classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3, gamma=0.0,
              learning_rate=0.05, max_delta_step=0, max_depth=3,
              min_child_weight=7, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)


# In[ ]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,X,y,cv=10)


# In[ ]:


score


# In[ ]:


score.mean()

