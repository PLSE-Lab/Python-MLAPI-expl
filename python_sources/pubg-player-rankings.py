#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **DATA PREPROCESSING**
# 1. Reading the data into the training dataframe
# 2. Dropping the NaN values and check if there are any nulls in dataset.
# 3. Seperating the features from target variable
# 4. Dropping the Unneccessary columns(features) like match id, id and Group id.

# In[ ]:


train_dataset = '../input/train_V2.csv'
test_dataset = '../input/test_V2.csv'


# In[ ]:


train_df = pd.read_csv(train_dataset)
train_df.head()


# In[ ]:


train_df.isnull().sum() #Check if there are any null
train_df = train_df.dropna() #remove nulls from datasets


# In[ ]:


target = train_df['winPlacePerc'] #target variable to find
features = train_df.drop(['winPlacePerc'],axis=1) #input features
features.head()


# In[ ]:


refine_features = features.drop(['Id','groupId','matchId'],axis=1) #drop unnecessary features
refine_features.info()


# **Visualization**
# 
# Plot a bar graph where we see the correlation between the target variable(winPlacePerc) and features to get a clear picture which feature is most correlated with the winning place percentage in a sorted ascending order.

# In[ ]:


train_df.corr()['winPlacePerc'].sort_values().plot(kind='bar',figsize=(11,7))


# In[ ]:


#Convert categorical variables to numerical by encoding of 0 and 1
refine_features = pd.get_dummies(refine_features)
refine_features.info()


# **Variance Inflation Factor(VIF) and Multicollinearity**
# 
# **Multicollinearity** - occurs when independent variables in a regression model are correlated. If this happens it can cause accuracy issues while fitting the model because independent variables should be independent and not correlated.
# 
# **VIF** - explains the amount of multicollinearity exists between independent variables(predictors) in regression analysis.
# So high VIF for a feature means highly correlated to other features and vice versa.
# When a VIF is above 2.5 then you cannot ignore multicollinearity.

# In[ ]:


#calculate vif of each column(feature)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['vif factor'] = [variance_inflation_factor(refine_features.values,i) for i in range(refine_features.shape[1])]
vif['features'] = refine_features.columns
vif.sort_values(by=['vif factor'],ascending=False)


# In[ ]:


#dropping columns based on hig VIF
list_of_drop_cols = ['maxPlace','numGroups','matchType_squad-fpp','matchType_duo-fpp','winPoints','matchType_solo-fpp','rankPoints','matchType_squad','matchType_duo','matchType_solo']
refine_features = refine_features.drop(list_of_drop_cols,axis = 1)
refine_features.shape


# In[ ]:


#Create cross validation test sets to check if model is trained well or not.
from sklearn.model_selection import train_test_split
Xtrain,Xvalidation,Ytrain,Yvalidation = train_test_split(refine_features,target,test_size=0.25)
print(len(Xtrain))
print(len(Ytrain))
print(len(Xvalidation))
print(len(Yvalidation))


# In[ ]:


#GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.metrics import r2_score

linear = GradientBoostingRegressor(learning_rate = 1.0, n_estimators = 100, max_depth = 4)
linear.fit(Xtrain,Ytrain)
pred = linear.predict(Xvalidation)
print(r2_score(Yvalidation,pred))


# **Load the Testing Dataset**
# 
# 1. Load the testing dataset in the test dataframe.
# 2. drop all the unnecessary variables with high VIF and encode the categorical variables into numerical.
# 3. Apply the gradient boosting Regressor to the testing dataframe.
# 4. Create the pred_df framework as per the submission file and save the file to csv.
# 

# In[ ]:



test_df = pd.read_csv(test_dataset)
test_features = test_df.drop(['Id','groupId','matchId'],axis = 1)
test_features = pd.get_dummies(test_features)
test_features = test_features.drop(['maxPlace','numGroups','matchType_squad-fpp'
                             ,'matchType_duo-fpp','winPoints','matchType_solo-fpp','rankPoints',
                             'matchType_squad','matchType_duo','matchType_solo'],axis=1)

Xtest = test_features
test_features.shape


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor


linear = GradientBoostingRegressor()
linear.fit(Xtrain,Ytrain)
pred = linear.predict(Xtest)
pred[:10]


# In[ ]:


pred_df = pd.DataFrame(pred,test_df['Id'],columns=['WinPlacePerc'])
pred_df.head()


# In[ ]:




pred_df.to_csv('sample_submission.csv')


# In[ ]:




