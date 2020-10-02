#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier, plot_importance

import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# #Import datasets

# In[ ]:


train = pd.read_csv("../input/learn-together/train.csv", index_col='Id')
test = pd.read_csv("../input/learn-together/test.csv", index_col='Id')

y = train['Cover_Type'] # this is the target
X = train.drop('Cover_Type', axis = 1)
X_test = test.copy()

print('Train set shape : ', X.shape)
print('Test set shape : ', X_test.shape)


# Note : large difference between train and test size. Will need to check input distributions.

# In[ ]:


X.head()


# In[ ]:


X_test.head()


# Check for data types and missing values

# In[ ]:


print('Missing Label? ', y.isnull().any())
print('Missing train data? ', X.isnull().any().any())
print('Missing test data? ', X_test.isnull().any().any())


# In[ ]:


print (X.dtypes.value_counts())
print (X_test.dtypes.value_counts())


# No missing data, everything in numeric. 
# Soil_type and Wilderness_area are categorial data already put as one hot encoded.

# In[ ]:


X.describe()


# In[ ]:


X.nunique()


# Soil_Type15 and Soil_Type7 have only one value. meaning these types of soils didnt appear in the training set.
# -> drop these column.
# 

# In[ ]:


X.drop(['Soil_Type15', 'Soil_Type7'], axis=1, inplace = True)
X_test.drop(['Soil_Type15', 'Soil_Type7'], axis=1, inplace = True)


# In[ ]:


X_test.describe()


# Many values have large numbers, std and means. Will need for scaling (ideally, we want normal distributions with (0,1))
# However, distributions are not very similar. 
# - Should we scale based on all data? test data only? train only?  (my intuition:on train only, need to check)
# - Do we need to scale binary data ?

# In[ ]:


columns = X.columns


# In[ ]:





# TODO:
# - Check and remove outliers
# -- Should we remove columns with too few data (e.g. in soil types)
# - Scale to normal-ish distributions
# - Check correlations and linearities

# #Model setup
# Try classic XGB

# In[ ]:


X_test_index = X_test.index # the scaler drops table index/columns and outputs simple arrays..
scaler = RobustScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


# In[ ]:


X_train,  X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=1)
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)


# In[ ]:


xgb= XGBClassifier( n_estimators=1000,  #todo : search for good parameters
                    learning_rate= 0.5,  #todo : search for good parameters
                    objective= 'binary:logistic', #this outputs probability,not one/zero. should we use binary:hinge? is it better for the learning phase?
                    random_state= 1,
                    n_jobs=-1)


# In[ ]:


xgb.fit(X=X_train, y=y_train,
        eval_metric='merror', # merror: Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases). 
        eval_set=[(X_val,y_val)],
        early_stopping_rounds = 100,
        verbose = False
       )
print(xgb.best_score)


# #checking some scores. Should to some real search
# - without scaler
# XGBClassifier( n_estimators=1000, learning_rate=1, objective= 'binary:logistic', random_state=1,  n_jobs=-1) # Validation split error : 0.165344  (early stop 20)
# XGBClassifier( n_estimators=1000, learning_rate=0.5, objective= 'binary:logistic', random_state=1,  n_jobs=-1) # Validation split error : 0.166336  (early stop 20)
# XGBClassifier( n_estimators=1000, learning_rate=0.1, objective= 'binary:logistic', random_state=1,  n_jobs=-1) # Validation split error : 0.215278  (early stop 20)
# - with scaler
# XGBClassifier( n_estimators=1000, learning_rate=1, objective= 'binary:logistic', random_state=1,  n_jobs=-1) # Validation split error : 0.165675  (early stop 20)
# XGBClassifier( n_estimators=1000, learning_rate=1, objective= 'binary:logistic', random_state=1,  n_jobs=-1) # Validation split error : 0.153439  (early stop 100)
# XGBClassifier( n_estimators=1000, learning_rate=0.5, objective= 'binary:logistic', random_state=1,  n_jobs=-1) # Validation split error : 0.153108 (early stop 100)
# XGBClassifier( n_estimators=1000, learning_rate=0.1, objective= 'binary:logistic', random_state=1,  n_jobs=-1) # Validation split error : 0.161045 (early stop 100)
# XGBClassifier( n_estimators=1000, learning_rate=0.1, objective= 'binary:logistic', random_state=1,  n_jobs=-1) # Validation split error : 0.21x  (early stop 20)
# 
# 

# In[ ]:


plt.figure(figsize=(25,10))
sns.barplot(y=xgb.feature_importances_, x=columns)


# #preparing for submission
# - fit model on the whole training set 
# - predict test set
# - format output

# In[ ]:


xgb.fit(X,y)
preds_test = xgb.predict(X_test)
preds_test.shape


# In[ ]:


preds_test


# In[ ]:


# Save test predictions to file
output = pd.DataFrame({'ID': X_test_index,
                       'TARGET': preds_test})
output.to_csv('submission.csv', index=False)
output.head()

