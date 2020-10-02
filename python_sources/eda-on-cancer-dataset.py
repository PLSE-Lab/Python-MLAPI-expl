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


# In[ ]:


train = pd.read_csv('../input/cancer/Train.csv')                                


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


train.info()


# In[ ]:


#As there are only N and R, Label encoding of Outcome column wont cause any problem
from sklearn.preprocessing import LabelEncoder

le= LabelEncoder()
train['Outcome']= le.fit_transform(train['Outcome'])


# In[ ]:


train['Outcome'].value_counts()


# In[ ]:


train=train.replace(['?'],'NaN')


# In[ ]:


train.isnull().sum()


# In[ ]:


train['Lymph_Node_Status'].value_counts()


# In[ ]:


#Since '0' is the mode for Lymph Node Status. We fill all NaN with '0'
train['Lymph_Node_Status']= train['Lymph_Node_Status'].replace('NaN', int(0))


# In[ ]:


train['Lymph_Node_Status']= train['Lymph_Node_Status'].astype(int)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

corr= train.corr()
f, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(corr,linewidths=.5, vmin=0, vmax=1,)


# In[ ]:


train.columns


# In[ ]:


train.iloc[0,:]


# In[ ]:


#Performing upsampling for R type of cancer
for i in range(0,198):
    if train['Outcome'][i]== 1:
        train = train.append(train.iloc[i,:], ignore_index= True)


# In[ ]:


from sklearn.utils import shuffle
train = shuffle(train)
train.reset_index(inplace=True, drop=True)


# In[ ]:


train.tail()


# In[ ]:


#Since the Areas and Perimeters are covered by radius, we'll drop them
drop= ['ID','perimeter_mean', 'area_mean','perimeter_std_dev', 'area_std_dev', 'Worst_perimeter', 'Worst_area']
train1 = train.drop(drop, axis= 1)
train = train.drop('ID', axis=1)


# In[ ]:


corr1= train1.corr()
f, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(corr1,linewidths=.5, vmin=0, vmax=1,)


# In[ ]:


from scipy.stats import skew, kurtosis
print(skew(train1['Time']))
print(kurtosis(train1['Time']))


#Inference-- Moderately skewed and Flat peak kurtosis which is not much problematic


# In[ ]:


plt.scatter(train['Tumor_Size'],train['Time'] )
#we cant see any trend between the two


# In[ ]:


plt.hist((train['Tumor_Size']), bins= 30)


# In[ ]:


train


# In[ ]:


#Checking outlier via Zscore

from scipy import stats
zscore= stats.zscore(train['Time']) 

for z in zscore:
    if z>=3 or z<=-3:
        print('True')

#Cannot see any outlier
   


# In[ ]:


from scipy.special import inv_boxcox
y,l= stats.boxcox(train['Time'])
print(inv_boxcox(y,l))


# # We will consider that time is not given while building the classifier as practically we cant tell the time until we dont know whether the type of cancer is N or R

# In[ ]:


time1= train1['Time']
target1= train1['Outcome']
train1_ot= train1.drop(['Outcome','Time'], axis=1)
train_ot= train.drop(['Outcome','Time'], axis=1)


# In[ ]:


from sklearn.preprocessing import RobustScaler as rs
train11=rs().fit(train1_ot).transform(train1_ot) #with dropped features like Area and Perimeter
train12= rs().fit(train_ot).transform(train_ot) #original dataset


# In[ ]:


from sklearn.model_selection import GridSearchCV as gsc,train_test_split as tts
from sklearn.ensemble import RandomForestClassifier as rfc


# In[ ]:


train_1,valtrain1,target_1,valtarget1=tts(train12,target1,test_size=0.2, random_state= 42)


# In[ ]:


'''gsc= gsc(estimator= rfc(), 
                  param_grid= {'max_depth': range(5,11), 
                              'n_estimators': (50,40,30),
                               'criterion' :['gini', 'entropy']
                              }, cv=5, scoring= 'f1'
                 , verbose=1, n_jobs= -1)
grid_result= gsc.fit(train_1, target_1)
best_params= grid_result.best_params_
print(best_params)'''


# In[ ]:


clf=rfc(criterion = 'gini', max_depth= 8, n_estimators= 50)

clf.fit(train_1, target_1)
pred= clf.predict(valtrain1)


# In[ ]:


from sklearn.metrics import f1_score, confusion_matrix, mean_squared_error as mse
print(confusion_matrix(valtarget1, pred))


# In[ ]:


from sklearn.model_selection import GridSearchCV as gsc
from xgboost import XGBClassifier
xgb = XGBClassifier(
    objective= 'binary:logistic',
    verbose=2, 
    subsample=0.6,
    nthread=4,
    seed=42
)
parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': (50,30, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}

grid_search = gsc(
    estimator=xgb,
    param_grid=parameters,
    scoring = 'f1',
    n_jobs = -1,
    cv = 7,
    verbose=True)
grid_result= grid_search.fit(train_1, target_1)
best_params= grid_result.best_params_
print(best_params)


# In[ ]:


from xgboost import XGBClassifier
xgb= XGBClassifier(learning_rate= 0.1, max_depth= 6, n_estimators=40, objective= 'binary:logistic',subsample=0.6)
xgb.fit(train_1,target_1)
pre= xgb.predict(valtrain1)
print(f1_score(valtarget1, pre))


# In[ ]:


print(confusion_matrix(valtarget1, pre))


# # While building regressor, we assume that we know the type of cancer (N/R) and that's why only Time is dropped below

# In[ ]:


train1_t= train1.drop(['Time'], axis=1)
train_t= train.drop(['Time'], axis=1)

