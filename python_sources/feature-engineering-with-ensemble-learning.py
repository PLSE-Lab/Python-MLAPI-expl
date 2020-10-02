#!/usr/bin/env python
# coding: utf-8

# # ExtraTrees with Feature Engineering 
# In this Tutorial I am going to explain in details how to use regression with Ensemble learning. In many cases machine learning algorithms don't perform well without feature engineering which is the process of filling NaNs and missing values , creating new features and etc... . I will also be performing some exploratory data analysis to perform feature engineering before implementing the suitable model.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Let's now explore the dataset that we have. First we will load the training set and test set and show all the columns using pandas.

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# The following two lines determines the number of visible columns and 
#the number of visible rows for dataframes and that doesn't affect the code
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


# # Data Exploration
# Let's now see how the training data and the test data look like

# In[ ]:


# Dataframe.head(n) function enables you to show the first n rows in the dataframe

train.head()


# In[ ]:


train.describe()


# In[ ]:


test.head()


# Let's now see how many data points we have for training.

# In[ ]:


print("The number of traning examples(data points) = %i " % train.shape[0])
print("The number of features we have = %i " % train.shape[1])


# Let's check if any of the columns contains NaNs or Nulls so that we can fill those values if they are insignificant or drop them. We may drop a whole column if most of its values are NaNs or fill its value according to its relation with other columns in the dataframe. Nones can also be 0 in some datasets and that is why i am going to use the describe of the train to see if the range of numbers is not reasonable or not. if you are dropping rows with NaNs and you notice that you need to drop a large portion of your dataset then you should think about filling the NaN values or drop a column that has most of its values missing.

# In[ ]:


train.describe()


# In[ ]:


train.isnull().sum()


# It seems we have some columns with most of the values null values, column Alley has 1369 null values out of 1460 total values and so as FireplaceQu, PoolQC, MiscFeature and Fence. I will remove those columns because filling it won't be an ease task as we can't find a relationship between most of them and the other features due to the huge numbers of missing values in those columns. 
# <lb>Let's now discover the correlation matrix for this dataset and see if we can combine features or drop some according to its correlation with the output labels after removing the mentioned columns.

# In[ ]:


train.drop(["PoolQC","PavedDrive",'Fence','MiscFeature','FireplaceQu','GarageFinish', 'Alley'], inplace = True, axis = 1 )
test.drop(["PoolQC","PavedDrive",'Fence','MiscFeature','FireplaceQu','GarageFinish' , 'Alley' ], inplace = True, axis = 1 )


# In[ ]:


train.head()


# 

# In[ ]:


train = pd.get_dummies(train)
test = pd.get_dummies(test)


# now we get the label from the training set and put it in a seperate series then we drop it from the training set for future use.

# In[ ]:


y = train['SalePrice']
train = train.drop(["SalePrice"], axis = 1)


# ### NOTE: 
# When using get dummies it is important to notice that it may give different results when applied on training other than when applied on test sets
# that is because if you have training = | c1 , c2 |           and test = | c1 , c2 |          then you will get new columns in training named [ a, b, c, d] and only in test [a, b]
#                                                                  |a ,    b  |                              |a ,    b  |
#                                                                  |c ,    d  |
# That is why we will removing columns in training sets that are not in test set

# In[ ]:


missing_cols = set( train.columns ) - set( test.columns )
print( missing_cols )


# In[ ]:


train.drop( missing_cols , inplace = True, axis = 1 )


# In[ ]:


train.head()


# We will fill NAN values in the rest of the columns using the mean values

# In[ ]:


mean = train.mean().astype(np.int32)
train.fillna( mean , inplace = True)

mean = test.mean().astype(np.int32)
test.fillna( mean , inplace = True)


# Now we will check again to see if all the NaN values have been removed

# In[ ]:


train.isnull().sum()


# Since all the NaNs were removed, We should now check the correlation matrix to see how features relate to the target feature.

# In[ ]:


import seaborn as sns

import matplotlib.pyplot as plt


corr = train.corr()
f, ax = plt.subplots(figsize=(25, 25))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5)


# Based on the relations between features, I will create new features to increase the accuracy of my tree-based model

# In[ ]:


train['LotFA'] = train['LotArea'] + train['LotFrontage']
train['LotFA2'] = train['LotArea'] - train['LotFrontage']

train['OverallRate'] = train['OverallQual'] + train['OverallCond']
train['OverallRate2'] = train['OverallQual'] - train['OverallCond']

train['yearAvg'] = train['YearBuilt'] + train['YearRemodAdd']
train['yearAvg2'] = train['YearBuilt'] - train['YearRemodAdd']

train['fsFeet'] = train['1stFlrSF'] + train['2ndFlrSF']
train['fsFeet2'] = train['1stFlrSF'] - train['2ndFlrSF']

train['bath'] = train['BsmtFullBath'] + train['BsmtHalfBath']
train['bath2'] = train['BsmtFullBath'] - train['BsmtHalfBath']

train['areaPerCar'] = train['GarageArea'] / train['GarageCars']

train['OpenEnclosedPorch'] = train['OpenPorchSF'] + train['EnclosedPorch']
train['OpenEnclosedPorch2'] = train['OpenPorchSF'] - train['EnclosedPorch']

train['yearSANEG'] = (train['YrSold']**2 - train['yearAvg']**2)**0.5
train['yearSAPOS'] = (train['YrSold']**2 + train['yearAvg']**2)**0.5


test['LotFA'] = test['LotArea'] + test['LotFrontage']
test['LotFA2'] = test['LotArea'] - test['LotFrontage']

test['OverallRate'] = test['OverallQual'] + test['OverallCond']
test['OverallRate2'] = test['OverallQual'] - test['OverallCond']

test['yearAvg'] = test['YearBuilt'] + test['YearRemodAdd']
test['yearAvg2'] = test['YearBuilt'] - test['YearRemodAdd']

test['fsFeet'] = test['1stFlrSF'] + test['2ndFlrSF']
test['fsFeet2'] = test['1stFlrSF'] - test['2ndFlrSF']

test['bath'] = test['BsmtFullBath'] + test['BsmtHalfBath']
test['bath2'] = test['BsmtFullBath'] - test['BsmtHalfBath']

test['areaPerCar'] = test['GarageArea'] / test['GarageCars']

test['OpenEnclosedPorch'] = test['OpenPorchSF'] + test['EnclosedPorch']
test['OpenEnclosedPorch2'] = test['OpenPorchSF'] - test['EnclosedPorch']

test['yearSANEG'] = (test['YrSold']**2 - test['yearAvg']**2)**0.5
test['yearSAPOS'] = (test['YrSold']**2 + test['yearAvg']**2)**0.5


# In[ ]:


train.fillna( 0 , inplace = True)
test.fillna( 0 , inplace = True)


# In[ ]:


train.isnull().sum()


# In[ ]:


from sklearn.cluster import KMeans
cluster = KMeans(n_clusters= 800, max_iter=300, tol=0.0001, verbose=0, random_state = 0, n_jobs=-1)


# In[ ]:


kmeans_train = cluster.fit(train)
labels_train = kmeans_train.labels_
labels_train


# In[ ]:


kmeans_testing = cluster.predict(test)
kmeans_testing


# In[ ]:


train['cluster'] = labels_train
test ['cluster'] = kmeans_testing


# Let's now check again the dimentions of our training set after engineering the catagorical features using the get_dummies function.

# In[ ]:


print("The number of traning examples(data points) = %i " % train.shape[0])
print("The number of features we have = %i " % train.shape[1])


# In[ ]:


id = test['Id']
test.drop(['Id'], axis = 1, inplace = True)
train.drop(['Id'], axis = 1, inplace = True)


# I will use Train test split from Scikit-learn to get an estimate for the result before submission. I used a very small test set because the data is too small.

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split( train.values , y.values, test_size=0.05, random_state=42 )


# # Fitting the Regression Model

# Now let's fit the model and using Random forests, ExtraTreesRegressor and support vector regressor ensemble learning using [Stacking regressor](https://rasbt.github.io/mlxtend/user_guide/regressor/StackingCVRegressor/) with the lasso regressor as meta regressor

# In[ ]:


from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor
from mlxtend.regressor import StackingCVRegressor

from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error  
from sklearn.ensemble import ExtraTreesRegressor

et  = ExtraTreesRegressor(n_estimators=950 ,  max_features = 'auto', max_leaf_nodes=None, n_jobs= -1, random_state = 0, verbose = 0)
gbr = GradientBoostingRegressor()
lasso = Lasso()
xgbr = XGBRegressor()
svr = SVR(kernel= 'rbf', gamma= 'auto', tol=0.001, C=100.0, max_iter=-1)
rf = RandomForestRegressor(n_estimators=900,  random_state=0)
lr = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=-1)
knnR = KNeighborsRegressor(n_neighbors=20, n_jobs=-1)
reg = StackingCVRegressor(regressors=[  lasso , xgbr , et],meta_regressor=lasso)

reg.fit(x_train, y_train, groups = None)


# We can check the score of the model on the x_test before predicting the output of the test set

# In[ ]:


reg.score(x_test,y_test)


# Now i will make sure that the traing and the test set cols are in the same order

# In[ ]:


test = test[train.columns]


# Now we will predict the output and making an output CSV

# In[ ]:


sub = pd.DataFrame({"Id": id ,"SalePrice": (reg.predict(test.values)).round(decimals=2)})
sub.to_csv("stackcv_linearsvc.csv", index=False) 


# In[ ]:


sub.head(10)

