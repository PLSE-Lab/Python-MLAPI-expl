#!/usr/bin/env python
# coding: utf-8

# In this kernel, I would visualise some of the Melbourne Housing data and model the data using Light GBM, and for comparison, a multi-linear model.
# 
# Version 4 includes postcode as a categorical variable and correcting some spelling mistakes.
# 
# ## Import Libaries:

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, cross_val_score

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from pandas.api.types import is_string_dtype

import re

#Models
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import lightgbm as lgbm


# ## Import data:
# 
# I am only going to use part of the data with prices.

# In[ ]:


df = pd.read_csv('../input/Melbourne_housing_FULL.csv')
df = df[df.Price.notnull()]

df.head()


# In[ ]:


df.info()


# ## Imputing Missing Data
# 
# Lets' take a look at the missing data:

# In[ ]:


df.isnull().sum()


# There are many lines of data with missing data.
# 
# -For the numerical values, we can impute them by mean (Bedroom2, Bathroom, Car, Landsize, BuildingArea, YearBuilt)
# -For Geographic data, we can look up the values by the addresses(Distance, Postcode, CouncilArea, Lattitude, Longtitude, Regionname, and Propertycount).
# 
# ### Imputing numerical missing values:

# In[ ]:


imp = SimpleImputer(strategy = 'mean')
#Bedroom2
imp.fit(df[['Bedroom2']])
df['Bedroom2'] = imp.transform(df[['Bedroom2']]).ravel()
#Bathroom
imp.fit(df[['Bathroom']])
df['Bathroom'] = imp.transform(df[['Bathroom']]).ravel()
#Car
imp.fit(df[['Car']])
df['Car'] = imp.transform(df[['Car']]).ravel()
#Landsize
imp.fit(df[['Landsize']])
df['Landsize'] = imp.transform(df[['Landsize']]).ravel()

#BuildingArea
imp.fit(df[['BuildingArea']])
df['BuildingArea'] = imp.transform(df[['BuildingArea']]).ravel()
#YearBuilt
imp.fit(df[['YearBuilt']])
df['YearBuilt'] = imp.transform(df[['YearBuilt']]).ravel()

df.isnull().sum()


# In[ ]:





# > ### The 3 Odd Home Sales

# In[ ]:


df[df['CouncilArea'].isnull()]


# In[ ]:


#Filling in data based on values from the same suburb

df.loc[29483,['Suburb']]='Fawkner'
df.loc[29483,['CouncilArea']]='Moreland City Council'
df.loc[29483,['Regionname']]='Northern Metropolitan'
df.loc[29483,['Propertycount']]= 5070
df.loc[29483,['Distance']]=12.4
df.loc[29483,['Postcode']]= 3060


df.loc[26888,['CouncilArea']]='Boroondara City Council'
df.loc[26888,['Regionname']]='Southern Metropolitan'
df.loc[26888,['Propertycount']]= 8920


df.loc[18523,['CouncilArea']]='Maribyrnong City Council'
df.loc[18523,['Regionname']]='Western Metropolitan'
df.loc[18523,['Propertycount']]= 7570


# In[ ]:


df.isnull().sum()


# In[ ]:


df_cleaned = df.drop(['Lattitude','Longtitude'], axis = 1)
df_cleaned.info()


# 

# ## Feature Engineering

# #### Creating the Street - Suburb Column

# In[ ]:


str(df.loc[1,['Address']].values)


# In[ ]:


a = re.split('\s', str(df.loc[1,['Address']].values))


# In[ ]:


b = a[1:]
type(b)


# In[ ]:


c = ' '.join(b)
c


# In[ ]:


def street_name(addr):
    a = re.split('\s', addr)
    b = a[1:]
    c = ' '.join(b)
    return c


# In[ ]:


df_cleaned['Street'] = df_cleaned['Address'].apply(street_name)
df_cleaned['Street'].tail(20)


# In[ ]:


df_cleaned['Street_Suburb'] = df_cleaned['Street']+'-'+df_cleaned['Suburb']
df_cleaned['Street_Suburb'].head(10)


# > ### Convert Postcode, Suburb, Street_Suburb, Type , CouncilArea, Method, SellerG, and Regionname to Categorical data

# In[ ]:


categories = ['Postcode', 'Street_Suburb','Type', 'CouncilArea', 'Method','Suburb', 'SellerG', 'Regionname']

for i in categories:
    df_cleaned[i+'_code'] = df_cleaned[i].astype('category').cat.codes


df_cleaned.head(10)


# ### Final preparation for machine learning:

# In[ ]:


df_train = df_cleaned.drop(categories, axis = 1)
df_train = df_train.drop(['Address', 'Street'], axis = 1)
df_train['Date'] = df_train['Date'].apply(pd.Timestamp)

#Converting Dates as days from 1st of January 2016 and dropping Date column
df_train['DaysFromJan2016']= df_train['Date']-pd.Timestamp('01-01-2016')
df_train['DaysFromJan2016']= df_train['DaysFromJan2016'].dt.days
df_train = df_train.drop('Date', axis = 1)
df_train.head()


# ### Train- Test Split:

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df_train.drop('Price', axis = 1), df_train['Price'], test_size = 0.3, random_state = 100)


# ## Using Light GBM to predict prices

# In[ ]:




#Create training dataset, clearly labeling X and y
train = lgbm.Dataset(X_train, label = y_train)

#Setting out the Parameter Dictionary
parameters = {}
parameters['learning_rate'] = 0.05 #learning rate - controlling how fast estimates change
parameters['boosting_type'] = 'gbdt' # for  traditional Gradient Boosting Decision Tree
parameters['objective'] = 'mae' #Minimising mean absolute error
parameters['metric'] = 'mae' 
parameters['feature_faction'] = 0.8 # LightGBM to select 80% of features for each tree
parameters['max_depth'] = 10
parameters['min_data'] = 10

model_lgbm = lgbm.train(parameters, train, 200) #Training for 200 iterations


# > ### Cross Validation

# In[ ]:


cv_results = lgbm.cv(parameters, train, num_boost_round=200, nfold=5, metrics = 'mae', 
                    verbose_eval=20, early_stopping_rounds=10)


# In[ ]:


#Display Results:
print('Current parameters:\n', parameters)
print('Best CV score (mean absolute error):', cv_results['l1-mean'][-1])


# ### Plotting the Results vs Testing Data

# In[ ]:


predictions = model_lgbm.predict(X_test)


# In[ ]:


plot = sns.jointplot(x= y_test, y= predictions, kind='reg', xlim=(0,10000000), joint_kws={'line_kws':{'color':'red'}})
plot.set_axis_labels(xlabel='Actual Price', ylabel='Predicted Price')
plt.figure(figsize = [15,15])


# ### Feature Importance:
# 
# The most important features by gain - the average training loss reduction gained when using a feature for splitting.

# In[ ]:


lgbm.plot_importance(model_lgbm, height = 0.7, title = 'Feature Importance', xlabel = 'Average Training Loss Reduction', importance_type = 'gain', max_num_features = 20)


# #### The most important features affecting house prices in Melbourne are:
# - Distance from CBD
# - Postcode
# - Dwelling type
# - Number of rooms
# 
# Distance from CBD is by far the most important feature.

# In[ ]:




