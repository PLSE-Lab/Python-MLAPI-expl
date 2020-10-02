#!/usr/bin/env python
# coding: utf-8

# ### Goal
# It is your job to predict the sales price for each house. For each Id in the test set, you must predict the value of the SalePrice variable. 
# 
# ### Metric
# Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)

# For this competition, I will use Ridge Regression to predict the house price. The reason I chose Ridge regression is because  Ridge regression will overcome the overfitting problem since we have  many features.
# 
# The below helps improve the validaiton root mean square error and final prediction score.
# 1.  Log transform the y variable since the evaluation is based on the RMSE of the logarithm of the predicted house and observed sales price (Improved prediction performance a lot, competition score from 0.14705 to 0.11959)
# 1.  Exclude outliers (Improved prediction performance). 
# 1. Get rid of multicollinearity features (Improved prediction performance).  
# 
# Belows are the things I have tried but did not help with the prediction performance
# 
# 1.  Remove the 10 features with smallest absolute Pearson Correlation coefficient (Prediction performance gets worse)
# 1.  Use the RidgeCV instead of the simple train/validaiton split to choose regularization parameter (Prediction performance stayed the same)
# 1.  Use Lasso and LassoCV (Prediction performance gets a little bit worse)
# 1. Use the StandardScaler to the numeric features makes the performance worse, score dropped from 0.11959 to 0.12037. For redige regression, set normalize=True also has worse performance.
# 1. Use principal component regression. The reason is that the variance could not be explained by the first few principal components. 
# 
# Next I will try random forest regression, support vector machine to build regression model, and use GridSearchCV to tune hyperparameters.

# * By using some simple techniques, we have get rank of 16%. I am still working on improving this kernel. I will keep updating my tries and whether they work or not. 
# * If you think my kernel is helpful, please give me a voteup. This is very important for new people like me. Thank you in advance.
# * If you have any question, please feel free to leave me a message, I will check every day. Thank you so much.

# ## Part I: Import the libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style("whitegrid")
import scipy.stats as stats


# ## Part II: Import the raw data and check data

# In[ ]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# In[ ]:


# 2.1 Check number of observations and columns
print(train.shape)
print(test.shape)
# Train has one more column the predictor variable SalePrice than the test data
print(set(train.columns)-set(test.columns))


# In[ ]:


# 2.2 Check the number of observations, data types and NAN for each variable
print(train.info())
print('-----------------------'*3)
print(test.info())


# In[ ]:


# 2.3 Check the descriptive info for 37 numeric variables, exclude Id
train.drop(['Id'], axis=1).describe()


# In[ ]:


# Check how many numeric columns and how many object columns, 38 numeric columns including ID
# 43 character columns
print(train.select_dtypes(exclude=['object']).shape)
print(train.select_dtypes(include=['object']).shape)


# In[ ]:


# 2.4 Check the first 5 observations and last five observations of train and head
#print(train.head(5))
print(train[:5])
print('/n')
print(train.tail(5))
#print(train[-5:])


# ## Part 3. Exploratory check relationship between the dependent variable and independent variable and outliers

# * Check multicollinearity
# * Check outliers

# * 3.1 Use heatmap to check the correlation between all numeric variables

# In[ ]:


plt.figure(figsize=(13,10))
sns.heatmap(train.corr(), vmax=0.8)


# From the above plot, there are several features highly correlated. These will cause multicollinearity. We need to drop one of them.
# * YearBuilt and GarageYrBlt, this is reasonable since many times YearBuilt and GarageYrBlt will be the same. Drop GarageYrBlt
# * GrLivArea and TotRmsAbvGrd, drop TotRmsAbvGrd
# * 1stFlrSF and TotalBsmtSF, drop TotalBsmtSF
# 

# In[ ]:


train.drop(['GarageYrBlt','TotRmsAbvGrd','TotalBsmtSF'], axis=1, inplace=True)
test.drop(['GarageYrBlt','TotRmsAbvGrd','TotalBsmtSF'], axis=1, inplace=True)


# * 3.2 Check the relationship between SalePrice and predictor variables

# In[ ]:


# Top 10 numeric features positively correlated with SalePrice
train.corr()['SalePrice'].sort_values(ascending=False).head(11)


# In[ ]:


train.corr()['SalePrice'].abs().sort_values(ascending=False).head(11)
top_corr_features=train.corr()['SalePrice'].abs().sort_values(ascending=False).head(11).index
top_corr_features


# In[ ]:


# Box-plot to check relationship between SalePrice and OverallQual
plt.figure(figsize=(10,7))
sns.boxplot(x='OverallQual', y='SalePrice', data=train)


# As we can see the better the overall quality of the house, the higher the salePrice which is very reasonable.

# In[ ]:


# Scatterplot to check the relationship between SalePrice and GrLivArea
plt.scatter(x='GrLivArea', y='SalePrice', data=train, color='r', marker='*')
train['GrLivArea'].sort_values(ascending=False).head(2)


# * From the scatter plot, there are two outliers with gross living area 5642, 4676 but the SalePrice is low. We will drop these two outliers by index

# In[ ]:


train.index[[523, 1298]]


# In[ ]:


print(train.shape)
train.drop(train.index[[523, 1298]], inplace=True)
print(train.shape)


# In[ ]:


print(top_corr_features)
box_feature=['SalePrice','OverallQual','GarageCars','FullBath', 'YearBuilt','YearRemodAdd','Fireplaces']
scatter_feature=['SalePrice', 'GrLivArea','1stFlrSF','GarageArea']
# Use sns.pairplot to check the relationship between the SalePrice and top 10 correlated features
sns.pairplot(train[scatter_feature])


# * I am thinking of better ways to visualize the relationship between the SalPrice and categorical features, any suggestion will be greatly appreciated.

# In[ ]:


sns.pairplot(train[box_feature], kind='scatter', diag_kind='hist')


# ## Part 4. Check missing values in features and impute

# In[ ]:


train.isnull().sum().sort_values(ascending=False)
# Check the NAN values as percentage
train_nan_pct=(train.isnull().sum())/(train.isnull().count())
train_nan_pct=train_nan_pct[train_nan_pct>0]
train_nan_pct.sort_values(ascending=False)


# * Drop features PoolQC, MiscFeature, Alley, Fence, FireplaceQu which has more than 40% of NAN and seems not very correlated with SalePrice
# * For other missing features, impute quantitative feature with median (since data are skewed from scatter plot). Impute category variable or qualitative variable with mode

# In[ ]:


train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace=True)
test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace=True)


# In[ ]:


train['GarageQual'].value_counts()


# In[ ]:


train_impute_index=train_nan_pct[train_nan_pct<0.3].index
train_impute_index
train_impute_mode=['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
train_impute_median=['LotFrontage', 'MasVnrArea']


# In[ ]:


# Impute character or qualitative feature with mode
for feature in train_impute_mode:
    train[feature].fillna(train[feature].mode()[0], inplace=True)
    test[feature].fillna(test[feature].mode()[0], inplace=True)


# In[ ]:


# Impute numeric feature with median
for feature in train_impute_median:
    train[feature].fillna(train[feature].median(), inplace=True)
    test[feature].fillna(test[feature].median(), inplace=True)


# In[ ]:


# There are no nan values in train
train.isnull().sum().sort_values(ascending=False).head(5)


# In[ ]:


test_only_nan=test.isnull().sum().sort_values(ascending=False)
test_only_nan=test_only_nan[test_only_nan>0]
print(test_only_nan.index)
test_impute_mode=['MSZoning', 'BsmtFullBath', 'Utilities','BsmtHalfBath', 'Functional', 'SaleType', 'Exterior2nd', 'Exterior1st', 'GarageCars', 'KitchenQual']
test_impute_median=['BsmtFinSF2','GarageArea', 'BsmtFinSF1','BsmtUnfSF' ]


# In[ ]:


# Impute test character feature with mode
for feature in test_impute_mode:
    test[feature].fillna(test[feature].mode()[0], inplace=True)
for feature in test_impute_median:
    test[feature].fillna(test[feature].median(), inplace=True)
#Impute test numeric feature with median


# In[ ]:


# Now there are no NAN values in both train and test data
test.isnull().sum().sort_values(ascending=False).head(5)


# ## Part 5. Combine train features and test features and create dummy variables for character features before runnning machine learning models

# In[ ]:


# Store the test data ID for competition purpose
TestId=test['Id']


# In[ ]:


total_features=pd.concat((train.drop(['Id','SalePrice'], axis=1), test.drop(['Id'], axis=1)))


# In[ ]:


total_features=pd.get_dummies(total_features, drop_first=True)
train_features=total_features[0:train.shape[0]]
test_features=total_features[train.shape[0]:]


# ## Part 6: Check the response variables

# In[ ]:


sns.distplot(train['SalePrice'])
# The response variable is right-skewed, we will log1p() transform y


# In[ ]:


train['Log SalePrice']=np.log1p(train['SalePrice'])
sns.distplot(train['Log SalePrice'])
# natural log one plus the array log(y+1) is more symmetric 


# In[ ]:


plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
sns.kdeplot(train['SalePrice'], legend=True)
plt.subplot(1,2,2)
sns.kdeplot(train['Log SalePrice'], legend=True)


# * We will compare the performance of buiding the model using y and log(y+1)

# ## Part 6: Train Validation split on the training data and Build Ridge Regression

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val=train_test_split(train_features, train[['SalePrice']], test_size=0.3, random_state=100)


# In[ ]:


# Import Ridge regression from sklearn
from sklearn.linear_model import Ridge
# Evaluate model performance using root mean square error
from sklearn.metrics import mean_squared_error
rmse=[]
# check the below alpha values for Ridge Regression
alpha=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

for alph in alpha:
    ridge=Ridge(alpha=alph, copy_X=True, fit_intercept=True)
    ridge.fit(X_train, y_train)
    predict=ridge.predict(X_val)
    rmse.append(np.sqrt(mean_squared_error(predict, y_val)))
print(rmse)
plt.scatter(alpha, rmse)


# In[ ]:


rmse=pd.Series(rmse, index=alpha)
rmse.argmin()


# In[ ]:


# Adjust alpha based on previous result
alpha=np.arange(8,14, 0.5)
rmse=[]

for alph in alpha:
    ridge=Ridge(alpha=alph, copy_X=True, fit_intercept=True)
    ridge.fit(X_train, y_train)
    predict=ridge.predict(X_val)
    rmse.append(np.sqrt(mean_squared_error(predict, y_val)))
print(rmse)
plt.scatter(alpha, rmse)
rmse=pd.Series(rmse, index=alpha)
print(rmse.argmin())


# In[ ]:


# Adjust alpha based on previous result
alpha=np.arange(10.5, 11.6, 0.1)
rmse=[]

for alph in alpha:
    ridge=Ridge(alpha=alph, copy_X=True, fit_intercept=True)
    ridge.fit(X_train, y_train)
    predict=ridge.predict(X_val)
    rmse.append(np.sqrt(mean_squared_error(predict, y_val)))
print(rmse)
plt.scatter(alpha, rmse)
rmse=pd.Series(rmse, index=alpha)
print(rmse.argmin())


# In[ ]:


# Use alpha=11.1 to predict the test data
ridge=Ridge(alpha=11.1)
# Use all training data to fit the model
ridge.fit(train_features, train[['SalePrice']])
predicted=ridge.predict(test_features)


# In[ ]:


submission=pd.DataFrame()
submission['Id']=TestId
submission['SalePrice']=predicted
submission.to_csv('submission.csv', index=False)


# ## Part 7 Use log(1+SalePrice)
# * By building a Ridge Regression with SalePrice on features, we achieve a score of 0.14705
# * Next we will build the Ridge model with the log(SalePrice+1) to check whether the prediction performance improves

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val=train_test_split(train_features, train[['Log SalePrice']], test_size=0.3, random_state=100)


# In[ ]:


# Import Ridge regression from sklearn
from sklearn.linear_model import Ridge
# Evaluate model performance using root mean square error
from sklearn.metrics import mean_squared_error
rmse=[]
# check the below alpha values for Ridge Regression
alpha=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

for alph in alpha:
    ridge=Ridge(alpha=alph, copy_X=True, fit_intercept=True)
    ridge.fit(X_train, y_train)
    predict=ridge.predict(X_val)
    rmse.append(np.sqrt(mean_squared_error(predict, y_val)))
print(rmse)
plt.scatter(alpha, rmse)
rmse=pd.Series(rmse, index=alpha)
print(rmse.argmin())
print(rmse.min())


# In[ ]:


# Adjust alpha based on previous result
alpha=np.arange(8,14, 0.5)
rmse=[]

for alph in alpha:
    ridge=Ridge(alpha=alph, copy_X=True, fit_intercept=True)
    ridge.fit(X_train, y_train)
    predict=ridge.predict(X_val)
    rmse.append(np.sqrt(mean_squared_error(predict, y_val)))
print(rmse)
plt.scatter(alpha, rmse)
rmse=pd.Series(rmse, index=alpha)
print(rmse.argmin())
print(rmse.min())


# In[ ]:


# Adjust alpha based on previous result
alpha=np.arange(10.5, 11.5, 0.1)
rmse=[]

for alph in alpha:
    ridge=Ridge(alpha=alph, copy_X=True, fit_intercept=True)
    ridge.fit(X_train, y_train)
    predict=ridge.predict(X_val)
    rmse.append(np.sqrt(mean_squared_error(predict, y_val)))
print(rmse)
plt.scatter(alpha, rmse)
rmse=pd.Series(rmse, index=alpha)
print('Minimum RMSE at alpah: ', rmse.argmin())
print('Minimum RMSE is: ', rmse.min())


# In[ ]:


# Use alpha=11 to predict the test data
ridge=Ridge(alpha=11)
# Use all training data to fit the model
ridge.fit(train_features, train[['Log SalePrice']])
predicted_log_price=ridge.predict(test_features)


# In[ ]:


# Transform back the log(SalePrice+1) to SalePrice
Test_price=np.exp(list(predicted_log_price))-1
Test_price


# In[ ]:


submission=pd.DataFrame()
submission['Id']=TestId
submission['SalePrice']=Test_price


# In[ ]:


submission.to_csv('submission.csv', index=False)

