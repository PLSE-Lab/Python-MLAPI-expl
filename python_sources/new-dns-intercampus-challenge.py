#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import some necessary librairies

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt  # Matlab-style plotting
from scipy import stats
from scipy.stats import norm, skew #for some statistics
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore warning (from sklearn and seaborn)

import os
print(os.listdir("../input"))


# In[ ]:


#Now let's import and put the train and test datasets in  pandas dataframe

train = pd.read_csv('../input/dnsprechallenge/train.csv')
test = pd.read_csv('../input/dnsprechallenge/test.csv')


# In[ ]:


##display the first five rows of the train dataset.
train.head(5)


# In[ ]:


##display the first five rows of the train dataset.
test.head(5)


# In[ ]:


#Lets check the number of features in our train and test set
print('Train size is {}'.format(train.shape))
print('Test size is {}'.format(test.shape))


# In[ ]:


#we'll drop some columns as they are unneccessary in the prediction process
id_cols = ['Product_Supermarket_Identifier']

train.drop(id_cols, axis=1, inplace=True)
test.drop(id_cols, axis=1, inplace=True)

print('Train size after dropping three columns is {}'.format(train.shape))
print('Test size after dropping three columns  is {}'.format(test.shape))


# **LET'S DO SOME EDA**
# 
# Product_Supermarket_Sales is the target variable we are trying to predict, so lets explore it.

# In[ ]:


# #Check the distribution
# sns.distplot(train['Product_Supermarket_Sales'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(train['Product_Supermarket_Sales'], plot=plt)


# In[ ]:


# #applying log transformation
# train['Product_Supermarket_Sales'] = np.log10(train['Product_Supermarket_Sales'])


# In[ ]:


# #Check the distribution
# sns.distplot(train['Product_Supermarket_Sales'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(train['Product_Supermarket_Sales'], plot=plt)


# In[ ]:


# #Lets plot some heatmap to find correlation among the features
# corrmat = train.corr()
# f, ax = plt.subplots(figsize=(5,4))
# sns.heatmap(corrmat, square=True)


# At a glance we can see that the highest correlated feature to Product_Supermaket_Sales is Product_Price followed by Supermaket_Opening_Year. And that makes sense because the more a shop sells  expensive goods the higher their total sales get.
# Another observation is that it seems the year of opening also has some correlation with product sales. Lets plot some one to one plot to see if this is a negative or positive trend .
# 

# In[ ]:


#Get percentage of missing data
train_missing = (train.isnull().sum() / len(train)) * 100
train_missing = train_missing.drop(train_missing[train_missing == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Percentage' : train_missing})
missing_data


# In[ ]:


#scatterplot of all features
cat_col = ['Product_Fat_Content','Product_Type','Supermarket_Location_Type','Supermarket_Type']
for col in cat_col: 
    sns.set()
    cols = ['Product_Identifier', 'Supermarket_Identifier',
           'Product_Fat_Content', 'Product_Shelf_Visibility', 'Product_Type',
           'Product_Price', 'Supermarket_Opening_Year',
           'Supermarket_Location_Type', 'Supermarket_Type',
           'Product_Supermarket_Sales']
    plt.figure()
    sns.pairplot(train[cols], size = 3.0, hue=col)
    plt.show()


# From the plot above, we can confirm that an increase in price of product really makes Total sales increase. Also there seems to be a very little trend in the Supermarket  opening year and the total sales, otherwise no other feature really correlates with Total sales

# **LET'S DO SOME FEATURE ENGINEERING**

# In[ ]:


#concatenate train and test sets
ntrain = train.shape[0]
ntest = test.shape[0]

#get target variable
y_train = train.Product_Supermarket_Sales.values

all_data = pd.concat((train,test)).reset_index(drop=True)
#drop target variable
all_data.drop(['Product_Supermarket_Sales'], axis=1, inplace=True)
print("Total data size is : {}".format(all_data.shape))


# **Let's take care of MISSING DATA**

# In[ ]:


#Get percentage of missing data
all_data_nan = (all_data.isnull().sum() / len(all_data)) * 100
all_data_nan = all_data_nan.drop(all_data_nan[all_data_nan == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Percentage' : all_data_nan})
missing_data


# So far we only have two columns with missing values.
# 
# >Since our features are small, I studied the mising columns and came up with the following conclusion for the supermarket_size column.
# 
# *  Grocery store in cluster 1 has a store size of small
# * Grocery store in cluster 3 has a store size of medium
# 
# * Supermarket_Type_2 store in cluster 3 has a store size of medium
# 
# * Supermarket_Type_3 store in cluster 3 has a store size of medium
# 
# * Supermarket_Type_1 store in cluster 1 has a store size of medium
# * Supermarket_Type_1 store in cluster 2 has a store size of small
# * Supermarket_Type_1 store in cluster 3 has a store size of high
# 
# This info will be used to fill the empty cells in Supermarket_size column.

# In[ ]:


def fill_nan_supermarket_size(mkt_type, mk_location, val):
    temp_df = all_data['Supermarket _Size'].loc[(all_data['Supermarket_Type']==  mkt_type ) & (all_data['Supermarket_Location_Type'] == mk_location)]
    temp_df.fillna(value=val, axis=0, inplace=True)
    all_data['Supermarket _Size'].loc[(all_data['Supermarket_Type']== mkt_type) & (all_data['Supermarket_Location_Type'] == mk_location)] = temp_df
    return 'Done'

    


# In[ ]:


# Fill all nan in Supermarket_size according to categories
fill_nan_supermarket_size('Grocery Store','Cluster 3', 'Medium')
fill_nan_supermarket_size('Supermarket Type3','Cluster 3', 'Medium')
fill_nan_supermarket_size('Supermarket Type2','Cluster 3', 'Medium')
fill_nan_supermarket_size('Supermarket Type1','Cluster 3', 'High')
fill_nan_supermarket_size('Supermarket Type1','Cluster 2', 'Small')
fill_nan_supermarket_size('Supermarket Type1','Cluster 1', 'Medium')


# In[ ]:


#Lets check the missing data percentage again
all_data_nan = (all_data.isnull().sum() / len(all_data)) * 100
all_data_nan = all_data_nan.drop(all_data_nan[all_data_nan == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Percentage' : all_data_nan})
missing_data


# 
# 
# We'll use the sklearn input function to take care of Product Weight

# In[ ]:


all_data.head()


# **Feature Engineering**

# In[ ]:


#Create the log version of product price
all_data['Product_Price_log'] = np.log1p(all_data['Product_Price'])
all_data['Product_Price_sqrt'] = np.sqrt(all_data['Product_Price'])
all_data['Product_Price_square'] = np.square(all_data['Product_Price'])


#Create some cross features
all_data['cross_Price_weight'] = all_data['Product_Price'] * all_data['Product_Weight']
all_data['cross_Price_visibility'] = all_data['Product_Price'] * all_data['Product_Shelf_Visibility']
all_data['cross_Price_visibility_weight'] = all_data['Product_Price'] * all_data['Product_Shelf_Visibility'] * all_data['Product_Weight']


# In[ ]:


all_data.head()


# In[ ]:





# In[ ]:


#change opening year to categories to remove 
train['Supermarket_Opening_Year'].unique()


# In[ ]:


#Supermarket size is a categorical feature.
dict_mkt_size = {'Small':1,'Medium':2,'High': 3}
dict_fat_content = {'Ultra Low fat': 1,'Low Fat': 2,'Normal Fat':3}
dict_year = {2005:'A', 1994:'B', 2014:'C', 2016:'D', 2011:'E', 2009:'F', 1992:'G', 2006:'H', 2004:'I'}

all_data['Supermarket _Size'] = all_data['Supermarket _Size'].map(dict_mkt_size)
all_data['Product_Fat_Content'] = all_data['Product_Fat_Content'].map(dict_fat_content)
all_data['Supermarket_Opening_Year'] = all_data['Supermarket_Opening_Year'].map(dict_year)


# In[ ]:


all_data.head()


# In[ ]:


X = pd.get_dummies(all_data)
print('All data size: ' + str(X.shape))


# In[ ]:


import random
#Lets get the new train and test set
train = X[:ntrain]
test = X[ntrain:]

#Let's shuffle our train set and labels
# random_indx = np.arange(ntrain)
# np.random.shuffle(random_indx)

# train = np.array(train)[random_indx]
# y_train = y_train[random_indx]

#Get the columns for importance plot
cols_4_imp = train.columns

print('Train size: ' + str(train.shape))
print('Test size: ' + str(test.shape))


# In[ ]:


train.to_csv("DSN_Supermarket_data_ensemble_train_4.csv", index=False)
test.to_csv("DSN_Supermarket_data_ensemble_test_4.csv", index=False)


# > **Now let's MODEL**

# In[ ]:


from sklearn.preprocessing import StandardScaler,Imputer
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
import lightgbm as lgb


# In[ ]:


imp = Imputer()
imp.fit(train)
train = imp.transform(train)
test = imp.transform(test)


# In[ ]:


#Scale features
scaler = RobustScaler()
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)


# In[ ]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train)
    rmse= np.sqrt(-cross_val_score(model, train, np.expm1(y_train), scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# In[ ]:


xgb1 = xgb.XGBRegressor(
 learning_rate =0.01,
 n_estimators=20000,
 max_depth=4,
 min_child_weight=8,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.1,
 nthread=4,
 scale_pos_weight=1,
 seed=27)


# In[ ]:


# print(rmsle_cv(xgb1))


# In[ ]:


# xgb1.fit(train,y_train)


# In[ ]:


# imp_feats = pd.DataFrame({'features':cols_4_imp,"importance": xgb1.feature_importances_})
# print(imp_feats)


# In[ ]:


# model_lgb = lgb.LGBMRegressor(learning_rate=0.01,n_estimators=5000)


# In[ ]:


# print(rmsle_cv(model_lgb))


# In[ ]:


# model_lgb.fit(train,y_train)


# In[ ]:


# avg_pred = ( 0.5 * xgb1.predict(test)) + (0.5 * model_lgb.predict(test))
# print("Mean Absolute Error : " + str(mean_absolute_error(avg_pred, y_test)))


# In[ ]:


# final = np.expm1(xgb1.predict(test))


# In[ ]:


# df = pd.read_csv('../input/dsn2018intercampus/SampleSubmission.csv')
# sub = df.drop('Product_Supermarket_Sales', axis=1)
# sub['Product_Supermarket_Sales'] = final


# In[ ]:


# sub.head()


# In[ ]:


# sub.to_csv('fe2_submission.csv', index=False)


# In[ ]:




