#!/usr/bin/env python
# coding: utf-8

# ## <center> Predicting House Price

# In[ ]:


#import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
import seaborn as sb
import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error

from scipy import stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p


# In[ ]:


#Read dataset
# df_train = pd.read_csv('data/train.csv')
# df_test = pd.read_csv('data/test.csv')

#Reading datasets train and test
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


#Review dataset
df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


#Review dataset size and shapes
df_train.shape, df_test.shape


# ### Data Back Up

# In[ ]:


#take a back up copy of dataset
df_train_copy = df_train.copy()
df_test_copy = df_test.copy()


# ### Handling Outliers

# In[ ]:


#Plot the outliers
plt.figure(figsize=(5,4))
plt.scatter(x=df_train['GrLivArea'], y=df_train['SalePrice'])
plt.xlabel('Ground Live Area')
plt.ylabel('SalePrice')
plt.show();


# In[ ]:


df_train.shape


# In[ ]:


#Delete the outliers
delidx = df_train[df_train['GrLivArea']>4000].index
df_train = df_train.drop(delidx, axis=0)
df_train.shape


# In[ ]:


#review outliers removal
plt.figure(figsize=(5,4))
plt.scatter(x=df_train['GrLivArea'], y=df_train['SalePrice'])
plt.xlabel('Ground Live Area')
plt.ylabel('SalePrice')
plt.show();


# ### Capture Total Number of rows in Train and Test

# In[ ]:


ntrain = df_train.shape[0]
ntest = df_test.shape[0]
ntrain, ntest


# ### Target Setup

# In[ ]:


#Target column is identified as House SalePrice
y_train = df_train['SalePrice'].values


# In[ ]:


#Drop column id as that do not contribute to calculate SalePrice
srs_testid = df_test['Id'] #Take backup of testid for final submission file

df_train.drop('Id', inplace=True, axis=1)
df_test.drop('Id', inplace=True, axis=1)

#Validate column reduction
df_train.shape, df_test.shape


# In[ ]:


#Visualize target data
plt.figure(figsize=(8,5))
sb.distplot(y_train, fit=norm)
plt.show();


# In[ ]:


#Since its not normalized taking log1p to normalize it
plt.figure(figsize=(8,5))
sb.distplot(np.log1p(y_train), fit=norm)
plt.show();


# In[ ]:


#As the graph now shown to be normalized, Hence transforming yvalue to log1p
y_train = np.log1p(y_train)


# ### Building All Data

# In[ ]:


#Concate both train and test dataset for data transformation
df_alldata = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

#Review all data
df_alldata.shape


# In[ ]:


#drop target column SalePrice from all data
df_alldata.drop('SalePrice', 
               inplace=True,
               axis=1)


# In[ ]:


df_alldata.head(3)


# ## Data Munging

# In[ ]:


#Find null values in dataset
alldata_na = df_alldata.isnull().sum()
alldata_na = alldata_na[alldata_na>0]
alldata_na = alldata_na.sort_values(ascending=False)
print('No of columns with nulls: ', len(alldata_na))


# ## Handling Missing Values

# In[ ]:


#For selected columns below impute missing values with 'None'
nonecols=['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','MasVnrType','MSSubClass']

for col in nonecols:
    df_alldata[col] = df_alldata[col].fillna('None')


# In[ ]:


#For selected columns below impute missing values with 0
zerocols = ['GarageYrBlt','GarageArea','GarageCars', 'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','MasVnrArea']

for col in zerocols:
    df_alldata[col] = df_alldata[col].fillna(0)


# In[ ]:


#For selected columns below fill null with mode
modecols=['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']

for col in modecols:
    df_alldata[col] = df_alldata[col].fillna(df_alldata[col].mode()[0])


# In[ ]:


#Drop the Utilities column
df_alldata.drop('Utilities', inplace=True, axis=1)


# In[ ]:


#Impute value 'Typ'
df_alldata['Functional'] = df_alldata.fillna('Typ')


# In[ ]:


#Impute lotfrontage null values with Neighbour hood median
df_alldata['LotFrontage'] = df_alldata.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


# In[ ]:


#Lookout for missing null values again
nas = df_alldata.isnull().sum()
nas = nas[nas>0]
nas


# In[ ]:


#Add new column TotalSF
df_alldata['TotalSF'] = df_alldata['TotalBsmtSF'] + df_alldata['1stFlrSF'] + df_alldata['2ndFlrSF']


# In[ ]:


#Drop noise columns
dropcols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
df_alldata.drop(dropcols, inplace=True, axis=1)


# In[ ]:


#Convert these cols to numerical
df_alldata['Functional'] = df_alldata['Functional'].astype(float)


# ---

# In[ ]:


#Selecting numerical features
num_feats = df_alldata.select_dtypes(exclude='object').columns
num_feats


# In[ ]:


catg_feats = df_alldata.dtypes[df_alldata.dtypes == 'object'].index
catg_feats


# In[ ]:


#Before onehot encoding
df_beforeonehot = df_alldata.copy()


# In[ ]:


df_alldata.shape


# In[ ]:


df_alldata_copy = df_alldata.copy()


# ## Perform One hot Encoding

# In[ ]:


#Actual onehot encoding avoiding dummy variable trap
for col in catg_feats:
    df_temp = df_alldata[col]
    df_temp = pd.DataFrame(df_temp)
    df_temp = pd.get_dummies(df_temp, prefix = col)
    temp = df_temp.columns[0] #Delete one dummy variable
    df_temp.drop(temp, inplace=True, axis=1)
    df_alldata = pd.concat([df_alldata, df_temp], axis=1).reset_index(drop=True)
    df_alldata.drop(col, inplace=True, axis=1) #Delete actual column from dataframe


# In[ ]:


df_alldata.shape


# ### Skew data to normalize feature values

# In[ ]:


skew_feats = df_alldata[num_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skew_feats = skew_feats[skew_feats>0.5]
print('No of Features to be Skewed: ', len(skew_feats))
print(skew_feats[:10])


# In[ ]:


plt.figure(figsize=(8,5))
skew_feats.plot(kind='bar')
plt.title('Skewed Features')
plt.xlabel('Features')
plt.ylabel('Skewed Value')
plt.show()


# In[ ]:


#Perform Box Cox Transformation on selected features having skew value > 0.5
Lambda=0.15
for col in skew_feats.index:
    df_alldata[col] = boxcox1p(df_alldata[col], Lambda)
    
print('No of Features Skewed: ',skew_feats.shape[0])


# ## Building Training and Test Set

# In[ ]:


df_train = df_alldata[:ntrain]
df_test = df_alldata[ntrain:]


# In[ ]:


df_train.shape


# In[ ]:


y_train.shape


# ### Building Model Validation Functions

# In[ ]:


#Defining cross validation strategy
cross_val = KFold(n_splits=10, shuffle=True, random_state=42)


# In[ ]:


#Define function to calculate rmse during training
def rmse_train(model, x, y):
    rmse = np.sqrt(-cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=cross_val, n_jobs=-1))
    return rmse.mean()


# In[ ]:


#Find rmse for prediction
def rmse_pred(y, y_pred):
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return rmse


# ### Building Function for Submission

# In[ ]:


def Submission(prediction):
    df_pred = pd.DataFrame({'Id':srs_testid, 'SalePrice':prediction})
    print('Sample Prediction:', prediction[:5])
    
    #Defining file name
    tday = datetime.today()
    tm = str(tday.date().day)+str(tday.date().month)+str('_')+str(tday.time().hour)+str(tday.time().minute)+str(tday.time().second)
    fn = 'Submission_'
    fn = str(fn)+str(tm)+str('.csv')
    
    #Saving prediction to csv
    df_pred.to_csv(fn, index=False)
    print('Submission file saved to', os.path.realpath(fn))


# ---

# ### Building Model

# In[ ]:


from sklearn.linear_model import LassoCV, RidgeCV, Lasso, Ridge, ElasticNet, ElasticNetCV
from sklearn.model_selection import KFold, train_test_split, ShuffleSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgbm


# In[ ]:


#Defining training inputs
X_train = df_train.values
y_train = y_train

#Define test inputs
X_test = df_test.values


# In[ ]:


alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 2, 10, 20, 50]


# In[ ]:


#Building Lasso Model
lcv = LassoCV(alphas=alphas, random_state=42, cv=cross_val, n_jobs=-1, max_iter=10000)


# In[ ]:


lcv.fit(X_train, y_train)


# In[ ]:


#Optimum alpha value for lasso model
lcv.alpha_


# In[ ]:


regressor_lasso = Lasso(alpha=0.0001, random_state=42)
regressor_lasso.fit(X_train,y_train)


# In[ ]:


#Review RMSE values for Lasso
print('Training RMSE:',rmse_train(regressor_lasso, X_train, y_train))


# In[ ]:


#Making prediction and review Test RMSE
print('Testing RMSE:',rmse_pred(y_train, regressor_lasso.predict(X_train)))


# In[ ]:


#Scoring Lasso prediction
pred = regressor_lasso.predict(X_test)
pred = np.expm1(pred)
Submission(pred)


# ## Secured score - 0.11976

# ### Building Ridge Model

# In[ ]:


alphas2 = [10, 12, 16, 12.5, 17, 10.001]


# In[ ]:


rcv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=cross_val)


# In[ ]:


rcv.fit(X_train, y_train)


# In[ ]:


rcv.alpha_


# In[ ]:


regressor_ridge = Ridge(alpha=10, max_iter=10000, random_state=42)


# In[ ]:


regressor_ridge.fit(X_train,y_train)


# In[ ]:


pred_ridge = regressor_ridge.predict(X_train)


# In[ ]:


#Review Train RMSE values
print('Training RMSE:',rmse_train(regressor_ridge, X_train, y_train))

#Making prediction and review Test RMSE
print('Testing RMSE:',rmse_pred(y_train, pred_ridge))


# In[ ]:


#Make Test Prediction
pred_ridge = regressor_ridge.predict(X_test)
pred_ridge = np.expm1(pred_ridge)
Submission(pred_ridge)


# ## Next Best Score - 0.11792

# In[ ]:


#Test both lasso and ridge
pred = regressor_lasso.predict(X_test)*0.5 + regressor_ridge.predict(X_test)*0.5
pred = np.expm1(pred)
Submission(pred)


# ### Secured score - 0.11814 (not an improvement)
