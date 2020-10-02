#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print("Shape of trainig set:",train.shape)
print("Shape of test set:",test.shape)


# In[ ]:


y = train['SalePrice']
total = train.append(test)
print(total.shape)
# Appended train data and test data for cleaning purpose


# In[ ]:


mask = total.isnull().sum()>0
print(sum(mask))
total.isnull().sum()


# In[ ]:


total.columns


# ## Univariate Analysis

# In[ ]:


print("mean of SalePrice", np.mean(y))
print("Median of Sale Price", np.median(y))
print("Minimum Sale Price", np.min(y))
print("Maximum Sale Price", np.max(y))


# In[ ]:


# Let's see the distribution
plt.figure(figsize=(10,8))
sns.distplot(y, label='Distribution of Sale Price')
plt.show()


# In[ ]:


# Let's plot the whiskers plot 
plt.figure(figsize=(10,8))
sns.boxplot(data=y)
plt.xlabel('Outlier detection from Sale Price')
plt.show()


# There are cetain outliers which can be filtered out by filtering data above 700000

# In[ ]:


# Let's find out how many outlier points are present
print('Number of possible outlier points above 600000=',sum(y>600000))
print('Number of possible outlier points above 700000=',sum(y>700000))


# So we may take 600000 or 700000 as outlier point after further analysis

# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(x='MSSubClass', data=train[train['SalePrice']<700000])
plt.show()


# 20 is most popularly sold and then 60 is sold. One common thing between them is that they represent 1945 and newer.

# In[ ]:


plt.figure(figsize=(8,6))
sns.countplot(x='MSZoning', data=train[train['SalePrice']<700000])
plt.show()


# In[ ]:


train['SalePrice'].groupby(train.MSZoning).median()


# In[ ]:


plt.figure(figsize=(8,6))
sns.countplot(x='SaleType', data=train[train['SalePrice']<700000])
plt.show()


# In[ ]:


plt.figure(figsize=(8,6))
sns.countplot(x='SaleCondition', data=train[train['SalePrice']<700000])
plt.show()


# In[ ]:


plt.figure(figsize=(8,6))
sns.countplot(x='BldgType', data=train[train['SalePrice']<700000])
plt.show()


# In[ ]:


plt.figure(figsize=(10,8))
sns.distplot(train['YearRemodAdd'])


# We can do the same for every variable but that would be too much. We will go for bivariate analysis for some most important variable. 

# ## Bivariate Analysis

# In[ ]:


plt.figure(figsize=(15,15))
corr = train.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# There are some obvious correlations like GarageArea is highly correlated to GarageCars, GarageYrBuilt and YearBuilt, OverallCond is correlated to yearBuilt, etc.

# In[ ]:


# Let's see how sales price varies with overall quality
plt.figure(figsize=(10,10))
ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,2)
sns.pointplot(x=train['OverallQual'],y=train['SalePrice'],ax=ax1)
sns.lineplot(x=train['OverallCond'],y=train['SalePrice'],ax=ax2)


# In[ ]:


sns.scatterplot(x=train.GrLivArea,y=train.SalePrice)


# Although, we can do more analysis, for now I will copy the data into temporary variables and move on to data preparation.

# ### Data Preparation

# In[ ]:


temp_y=y.copy()
temp_train=train.copy()
temp_test=test.copy()
# temp_train.drop('SalePrice', axis=1,inplace=True)
temp_total = temp_train.append(temp_test)
print("temp_y shape", temp_y.shape)
print("temp_train shape", temp_train.shape)
print("temp_test shape", temp_test.shape)
print("temp_total shape", temp_total.shape)


# We have now removed the skewed data and we can now apply the normalization if needed but since all values are in same range after log we can go without normalization.

# In[ ]:


temp_total.shape


# In[ ]:


# s = temp_total.isnull().sum()
# s[s>0]
sns.heatmap(temp_total.isnull())


# We shall preprocesses total data based on training data not on the basis of total data as that would make the data biased towards test set.

# In[ ]:


print(temp_train['MiscFeature'].value_counts())
# Lets just make Miscfeature such that if any value is present it is replaced as 1 otherwise 0
temp_total.loc[~temp_total['MiscFeature'].isnull(),'MiscFeature']=1
temp_total.loc[temp_total['MiscFeature'].isnull(),'MiscFeature']=0
print(temp_total['MiscFeature'].value_counts())


# In[ ]:


# Lets do the same for PoolQC and Alley
temp_total.loc[~temp_total['PoolQC'].isnull(),'PoolQC']=1
temp_total.loc[temp_total['PoolQC'].isnull(),'PoolQC']=0

print("Null PoolQC in trainig set:",temp_total['PoolQC'].isnull().sum())


# In[ ]:


# For Alley and Fence
temp_total.loc[~temp_total['Alley'].isnull(),'Alley']=1
temp_total.loc[temp_total['Alley'].isnull(),'Alley']=0

print("Null Alley in test set:",temp_total['Alley'].isnull().sum())

temp_total.loc[~temp_total['Fence'].isnull(),'Fence']=1
temp_total.loc[temp_total['Fence'].isnull(),'Fence']=0

print("Null Fence in trainig set:",temp_total['Fence'].isnull().sum())


# In[ ]:


# I tried including these features but dropiing is better option.
temp_total.drop('MiscFeature', axis=1,inplace=True)
temp_total.drop('PoolQC', axis=1,inplace=True)
temp_total.drop('Alley', axis=1,inplace=True)
temp_total.shape


# In[ ]:


s = temp_total.isnull().sum()
s[s>0]


# In[ ]:


(temp_total[temp_total['FireplaceQu'].isnull()])['Fireplaces'].value_counts()


# In[ ]:


# In FireplaceQu, null means no fireplace, so let's put NF for No-Fireplace
# Although same meant for above changed attributes, we changed them to binary values because they had too many null values
temp_total.loc[temp_total['FireplaceQu'].isnull(),'FireplaceQu']='NA'
plt.figure(figsize=(10,8))

sns.countplot(x='FireplaceQu',data=temp_total)

print("Null Fireplace in trainig set:",temp_total['FireplaceQu'].isnull().sum())


# In[ ]:


# Let's fill up the remaining null values
s = temp_total.isnull().sum()
s[s>0]


# In[ ]:


(temp_total[temp_total['GarageYrBlt'].isnull()])['GarageArea'].value_counts()


# If there is no garage there is a null value for all garage fields, so let's put NA to all such fields. Also there is an outlier with 360.0 which we convert to 0.0 because if there is no garage there should be no Garage Area which means it may be an artificial error(error in data collection, etc.)

# In[ ]:


# If there is no garage why should there be anything related to garage
temp_total.loc[temp_total['GarageYrBlt'].isnull(),'GarageCars']=0
temp_total.loc[temp_total['GarageYrBlt'].isnull(),'GarageArea']=0.0
temp_total.loc[temp_total['GarageYrBlt'].isnull(),'GarageQual']='NA'
temp_total.loc[temp_total['GarageYrBlt'].isnull(),'GarageCond']='NA'
temp_total.loc[temp_total['GarageYrBlt'].isnull(),'GarageFinish']='NA'
temp_total.loc[temp_total['GarageYrBlt'].isnull(),'GarageType']='NA'
temp_total.loc[temp_total['GarageYrBlt'].isnull(),'GarageYrBlt']=0
s = temp_total.isnull().sum()
s[s>0]


# In[ ]:


temp_total[temp_total['LotFrontage'].isnull()].head(10)


# In[ ]:


# There is some interesting pattern out there, let's make some plots.
plt.figure(figsize=(20,10))
ax1 = plt.subplot(3,3,1)
ax2 = plt.subplot(3,3,2)
ax3 = plt.subplot(3,3,3)
sns.countplot(x='MSZoning', data=temp_total[temp_total['LotFrontage'].isnull()],ax=ax1)
sns.countplot(x='Street', data=temp_total[temp_total['LotFrontage'].isnull()], ax=ax2)
sns.countplot(x='Utilities', data=temp_total[temp_total['LotFrontage'].isnull()], ax=ax3)


# In[ ]:


# Lets see what are mean and median for these values.
print((temp_train[temp_train['MSZoning']=='RL'])['LotFrontage'].mean())
print((temp_train[temp_train['MSZoning']=='RL'])['LotFrontage'].median())
print((temp_train[temp_train['Street']=='Pave'])['LotFrontage'].mean())
print((temp_train[temp_train['Street']=='Pave'])['LotFrontage'].median())
print((temp_train[temp_train['Utilities']=='AllPub'])['LotFrontage'].mean())
print((temp_train[temp_train['Utilities']=='AllPub'])['LotFrontage'].median())
print((temp_train[(temp_train['MSZoning']=='RL')&(temp_train['Street']=='Pave')&(temp_train['Utilities']=='AllPub')])['LotFrontage'].mean())
print((temp_train[(temp_train['MSZoning']=='RL')&(temp_train['Street']=='Pave')&(temp_train['Utilities']=='AllPub')])['LotFrontage'].median())


# Its clear that latter two properties are more likely. We will fill the missing values with mean of median value just to include all. Median is less sensitive to outliers, so we chose median. We use training set values so that we do not over optimize on test set.

# In[ ]:


meanForLotFrontage = np.mean([72.0,69.0,69.0])
# Not a good idea to hardcode the values but for now its okay.
print(meanForLotFrontage)
# Let's fill the missing values.
temp_total.loc[temp_total['LotFrontage'].isnull(),'LotFrontage']=meanForLotFrontage
temp_total['LotFrontage'].isnull().sum()


# In[ ]:


(temp_total[temp_total['BsmtQual'].isnull()])['TotalBsmtSF'].value_counts()
# All these should be filled with zero or NA because if there is no basement there should be no other value for basement.


# In[ ]:


# All remaining missing values are easy to fill, let's examine Basement properties where null means no basement
temp_total.loc[temp_total['BsmtQual'].isnull(),'BsmtCond']='NA'
temp_total.loc[temp_total['BsmtQual'].isnull(),'BsmtExposure']='NA'
temp_total.loc[temp_total['BsmtQual'].isnull(),'BsmtFinType1']='NA'
temp_total.loc[temp_total['BsmtQual'].isnull(),'BsmtFinType2']='NA'
temp_total.loc[temp_total['BsmtQual'].isnull(),'BsmtFinSF1']=0.0
temp_total.loc[temp_total['BsmtQual'].isnull(),'BsmtFinSF2']=0.0
temp_total.loc[temp_total['BsmtQual'].isnull(),'BsmtUnfSF']=0.0
temp_total.loc[temp_total['BsmtQual'].isnull(),'TotalBsmtSF']=0.0
temp_total.loc[temp_total['BsmtQual'].isnull(),'BsmtFullBath']=0.0
temp_total.loc[temp_total['BsmtQual'].isnull(),'BsmtHalfBath']=0.0
temp_total.loc[temp_total['BsmtQual'].isnull(),'BsmtQual']='NA'
s = temp_total.isnull().sum()
s[s>0]


# In[ ]:


print((temp_train[(temp_train['Street']=='Pave') & (temp_train['Utilities']=='AllPub') ])['MasVnrArea'].mean())
print((temp_train[(temp_train['Street']=='Pave') & (temp_train['Utilities']=='AllPub') ])['MasVnrArea'].median())
print((temp_train[(temp_train['Street']=='Pave') & (temp_train['Utilities']=='AllPub') ])['MasVnrArea'].value_counts())
(temp_train[(temp_train['Street']=='Pave') & (temp_train['Utilities']=='AllPub') ])['MasVnrType'].value_counts()


# In[ ]:


# Let's fill all other missing values
temp_total.loc[temp_total['MasVnrType'].isnull(),'MasVnrType']='None'
temp_total.loc[temp_total['MasVnrArea'].isnull(),'MasVnrArea']=0.0
temp_total.loc[temp_total['MSZoning'].isnull(), 'MSZoning']='RL'
temp_total.loc[temp_total['Utilities'].isnull(), 'Utilities']='AllPub'
temp_total.loc[temp_total['Exterior1st'].isnull(), 'Exterior1st']='VinylSd'
temp_total.loc[temp_total['Exterior2nd'].isnull(), 'Exterior2nd']='VinylSd'
temp_total.loc[temp_total['BsmtFinSF1'].isnull(), 'BsmtFinSF1']=0.0
temp_total.loc[temp_total['BsmtFinSF2'].isnull(), 'BsmtFinSF2']=0.0
temp_total.loc[temp_total['BsmtUnfSF'].isnull(), 'BsmtUnfSF']=0.0
temp_total.loc[temp_total['TotalBsmtSF'].isnull(), 'TotalBsmtSF']=0.0
temp_total.loc[temp_total['Electrical'].isnull(), 'Electrical']='SBrkr'
temp_total.loc[temp_total['BsmtFullBath'].isnull(), 'BsmtFullBath']=0.0
temp_total.loc[temp_total['BsmtHalfBath'].isnull(), 'BsmtHalfBath']=0.0
temp_total.loc[temp_total['BsmtFullBath'].isnull(), 'BsmtFullBath']=0.0
temp_total.loc[temp_total['KitchenQual'].isnull(), 'KitchenQual']='TA'
temp_total.loc[temp_total['Functional'].isnull(), 'Functional']='Typ'
temp_total.loc[temp_total['GarageCars'].isnull(), 'GarageCars']=2.0
temp_total.loc[temp_total['GarageArea'].isnull(), 'GarageArea']=480.0
temp_total.loc[temp_total['SaleType'].isnull(), 'SaleType']='WD'


# In[ ]:


# temp_total=temp_total.ffill(axis=0)
s=temp_total.isnull().sum()
s[s>0]


# In[ ]:


temp_total.loc[temp_total['BsmtCond'].isnull(),'BsmtCond']='TA'
temp_total.loc[temp_total['BsmtExposure'].isnull(),'BsmtExposure']='No'
temp_total.loc[temp_total['BsmtFinType2'].isnull(),'BsmtFinType2']='Unf'


# In[ ]:


s=temp_total.isnull().sum()
s[s>0]


# So there are no more null values left.
# Let's now deal with correlations and perform feature Engineering
# ### Feature Engineering

# In[ ]:


# Lets plot ditribution of garage area and if it is not skewed we shall drop GarageCars
# Otherwise we shall drop GarageArea
# We can keep the product of GarageArea and GarageCars
print(temp_total['GarageArea'].corr(temp_total['GarageCars']))
plt.figure(figsize=(10,8))
ax1=plt.subplot(2,2,1)
ax2=plt.subplot(2,2,2)
sns.distplot(temp_train['GarageArea'],ax=ax1)
sns.distplot(temp_train['GarageArea']*temp_train['GarageCars'],ax=ax2)


# In[ ]:


# Lets make a new feature by mutiplying the GarageCars and GarageArea
# We will create a new variable summing all areas
temp_total['TotalArea']=temp_total['TotalBsmtSF'] + temp_total['1stFlrSF'] + temp_total['2ndFlrSF'] + temp_total['GrLivArea'] +temp_total['GarageArea']
temp_total['Bathrooms'] = temp_total['FullBath'] + temp_total['HalfBath']*0.5+temp_total['BsmtFullBath']+0.5*temp_total['BsmtHalfBath'] 
print(temp_total.shape)
temp_total['Garage'] = temp_total['GarageArea']*temp_total['GarageCars']
# temp_total.drop(['GarageArea','GarageCars'],axis=1,inplace=True)
# temp_total.drop(['FullBath','HalfBath','BsmtHalfBath','BsmtFullBath'], axis=1, inplace=True)
print(temp_total.shape)


# We will keep working on feature engineering but for now let's proceed.

# ### Dummy variables

# In[ ]:


(temp_total.dtypes)


# In[ ]:


temp_total.columns


# In[ ]:


# These features need to be categorical instead.
# convert_features = {'LotFrontage':float,'LotArea':float,'YearBuilt':int,'YearRemodAdd':int,'MasVnrArea':float,'BsmtFinSF1':float,'BsmtFinSF2':float,'BsmtUnfSF':float,'TotalBsmtSF':float,'1stFlrSF':float,'2ndFlrSF':float,'LowQualFinSF':float,'GrLivArea':float,'TotalArea':float,'Bathrooms':float,'TotRmsAbvGrd':int,'Fireplaces':int,'GarageYrBlt':float,'Garage':float,'WoodDeckSF':float,'OpenPorchSF':float,'EnclosedPorch':float,'3SsnPorch':float,'ScreenPorch':float,'PoolArea':float}
# temp_total=temp_total.astype(convert_features)


# In[ ]:


# Some actually categorical features are misinterpreted as categorical
convert_features={
    'MSSubClass':str,
    'OverallCond':str,
    'OverallQual':str,
    'SaleCondition':str
}


# In[ ]:


temp_total=temp_total.astype(convert_features)


# In[ ]:


temp_total = pd.get_dummies(temp_total,drop_first=True)
temp_total.shape


# In[ ]:


temp_total.drop('Id',axis=True, inplace=True)


# We have significantly reduced number of features, earlier it was more than 350 but now its 275. Since we are using linear model, lesser dimensions helps.

# ### Skewness removal

# In[ ]:


plt.figure(figsize=(20,10))
ax1=plt.subplot(2,2,1)
ax2=plt.subplot(2,2,2)
print("GrLivArea skew:",temp_total['GrLivArea'].skew())
print("GrLivArea kurtosis:",temp_total['GrLivArea'].kurtosis())
sns.distplot(temp_total['GrLivArea'],ax=ax1)
print("LotArea skew:",temp_total['LotArea'].skew())
print("LotArea kurtosis:",temp_total['LotArea'].kurtosis())
sns.distplot(temp_total['LotArea'],ax=ax2)


# In[ ]:


plt.figure(figsize=(20,10))
ax1=plt.subplot(2,2,1)
ax2=plt.subplot(2,2,2)
print("1stFlrSF skew:",temp_total['1stFlrSF'].skew())
print("1stFlrSF kurtosis:",temp_total['1stFlrSF'].kurtosis())
sns.distplot(temp_total['1stFlrSF'],ax=ax1)
print("2ndFlrSF skew:",temp_total['1stFlrSF'].skew())
print("2ndFlrSF kurtosis:",temp_total['1stFlrSF'].kurtosis())
sns.distplot(temp_total['2ndFlrSF'],ax=ax2)


# In[ ]:


plt.figure(figsize=(20,10))
ax1=plt.subplot(2,2,1)
ax2=plt.subplot(2,2,2)
print("TotalArea skew:",temp_total['TotalArea'].skew())
print("TotalArea kurtosis:",temp_total['TotalArea'].kurtosis())
sns.distplot(temp_total['TotalArea'],ax=ax1)
print("Garage skew:",temp_total['Garage'].skew())
print("Garage kurtosis:",temp_total['Garage'].kurtosis())
sns.distplot(temp_total['Garage'],ax=ax2)


# In[ ]:


# Lets apply log1p transform to all of these
temp_total['GrLivArea']=np.log1p(temp_total['GrLivArea'])
temp_total['LotArea']=np.log1p(temp_total['LotArea'])
temp_total['1stFlrSF']=np.log1p(temp_total['1stFlrSF'])
temp_total['2ndFlrSF']=np.log1p(temp_total['2ndFlrSF'])
temp_total['TotalArea']=np.log1p(temp_total['TotalArea'])
temp_total['Garage']=np.log1p(temp_total['Garage'])


# ### Model Selection

# In[ ]:


temp_train = temp_total.iloc[:1460,:]
temp_test = temp_total.iloc[1460:,:]
print(temp_train.shape)
print(temp_test.shape)


# In[ ]:


temp_test.drop('SalePrice', axis=1,inplace=True)
temp_train=temp_train[temp_train['SalePrice']<700000]
temp_y = temp_train['SalePrice']
temp_train.drop('SalePrice', axis=1,inplace=True)
temp_y = np.log1p(temp_y)
print(temp_test.shape)
print(temp_train.shape)


# In[ ]:


sns.distplot(temp_y)


# In[ ]:


#For the purpose of preprocessing we need to divide the training set into validation set and train set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test = train_test_split(temp_train, temp_y,test_size=0.2, random_state=42)
print("X_train size:", X_train.shape)
print("X_test size:",X_test.shape)


# #### Scaling
# Using Robust Scalar

# In[ ]:


from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

temp_test=scaler.transform(temp_test)


# #### Let's start with Ridge Regression and Lasso Regression

# In[ ]:


from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


# In[ ]:


ridge=Ridge()
params = {'alpha':[i for i in range (1,50)]}
ridge_cv=GridSearchCV(estimator=ridge, param_grid=params, n_jobs=-1)
ridge_cv.fit(X_train,y_train)
print("The best Alpha is: ", ridge_cv.best_params_)
print("The best score achieved: ",np.sqrt(ridge_cv.best_score_))


# In[ ]:


# We obtained best ridge model with alpha=11. Let's keep this model for future use if any.
ridge = Ridge(alpha=0.25)
ridge.fit(X_train,y_train)
print("R2 score on training set", ridge.score(X_train,y_train))
pred_ridge_train = ridge.predict(X_train)
print("RMSE Ridge on training set", str(np.sqrt(mean_squared_error(y_train, pred_ridge_train))))
# On test set
pred_ridge_test = ridge.predict(X_test)
print("R2 score on test set", ridge.score(X_test,y_test))
print("RMSE Ridge on test set", str(np.sqrt(mean_squared_error(y_test, pred_ridge_test))))


# In[ ]:


lasso=Lasso()
params = {'alpha':[0.0001,0.0004,0.0009,0.01,0.1,1.0,10,100,1000]}
lasso_cv=GridSearchCV(estimator=lasso, param_grid=params, n_jobs=-1)
lasso_cv.fit(X_train,y_train)
print("The best Alpha is: ", lasso_cv.best_params_)
print("The best score achieved: ",np.sqrt(lasso_cv.best_score_))


# In[ ]:


# We obtained best ridge model with alpha=11. Let's keep this model for future use if any.
lasso = Lasso(alpha=0.000098, max_iter=1e7)
lasso.fit(X_train,y_train)
print("R2 score on training set", lasso.score(X_train,y_train))
pred_lasso_train = lasso.predict(X_train)
print("RMSE Lasso on training set", str(np.sqrt(mean_squared_error(y_train, pred_lasso_train))))
# On test set
pred_lasso_test = lasso.predict(X_test)
print("R2 score on training set", lasso.score(X_test,y_test))
print("RMSE Lasso on training set", str(np.sqrt(mean_squared_error(y_test, pred_lasso_test))))


# In[ ]:


# Elastic Net
from sklearn.linear_model import ElasticNetCV
alphas = [10,1,0.1,0.01,0.001,0.002,0.003,0.004,0.005]
l1_ratio=[0.1, 0.3,0.5, 0.9, 0.95, 0.99, 1]
elastic_cv = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratio, cv=15, max_iter=1e6)
elastic = elastic_cv.fit(X_train, y_train.ravel())
pred_elastic=elastic.predict(X_test)
print('RMSE test = ' + str(np.sqrt(mean_squared_error(y_test, pred_elastic))))
print(elastic_cv.alpha_)
print(elastic_cv.l1_ratio_)


# In[ ]:


from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet(alpha=0.0000985, l1_ratio=1.0, max_iter=1e7)
elastic = elasticnet.fit(X_train, y_train.ravel())
pred_elastic=elastic.predict(X_test)
print('RMSE test = ' + str(np.sqrt(mean_squared_error(y_test, pred_elastic))))


# We will use lasso or elastic net in final predictions

# In[ ]:


# from xgboost.sklearn import XGBRegressor

# xg_reg = XGBRegressor()
# xgparam_grid= {'learning_rate' : [0.01],'n_estimators':[2000, 3460, 4000],
#                                     'max_depth':[3], 'min_child_weight':[3,5],
#                                     'colsample_bytree':[0.5,0.7],
#                                     'reg_alpha':[0.0001,0.001,0.01,0.1,10,100],
#                                    'reg_lambda':[1,0.01,0.8,0.001,0.0001]}

# xg_grid=GridSearchCV(xg_reg, param_grid=xgparam_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
# xg_grid.fit(x_train,y_train)
# print(xg_grid.best_estimator_)
# print(xg_grid.best_score_)


# In[ ]:


from xgboost.sklearn import XGBRegressor
xgboost= XGBRegressor(base_score=0.5, booster='gbtree',gamma=0,
             importance_type='gain', learning_rate=0.01, max_delta_step=0,
             max_depth=3, min_child_weight=0.1, n_estimators=1000,
             n_jobs=-1, objective='reg:squarederror', random_state=0,
             reg_alpha=0.000098, reg_lambda=0.001)
xgb_model=xgboost.fit(X_train,y_train)
xgb_pred=xgb_model.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, xgb_pred)))


# In[ ]:


from sklearn.ensemble import VotingRegressor
vr = VotingRegressor([('Ridge', ridge),('Lasso',lasso),('ElasticNet',elastic),('XGBoost',xgb_model)])
voting_model = vr.fit(X_train, y_train.ravel())
vr_pred = voting_model.predict(X_test)
print("RMSE:",np.sqrt(mean_squared_error(vr_pred, y_test)))


# In[ ]:


from mlxtend.regressor import StackingRegressor

stack_reg = StackingRegressor(regressors=[elastic,ridge, lasso, voting_model], 
                           meta_regressor=xgb_model, use_features_in_secondary=True
                          )

stack_model=stack_reg.fit(X_train, y_train.ravel())
stacking_pred=stack_model.predict(X_test)
print("RMSE Stacking:", np.sqrt(mean_squared_error(stacking_pred, y_test)))


# In[ ]:


avg_test = (0.15*vr_pred+0.35*pred_lasso_test+0.15*stacking_pred+0.35*pred_ridge_test)
print("RMSE on averaged models:", np.sqrt(mean_squared_error(avg_test,y_test)))


# In[ ]:


# Let's train each of these models on full train set rather than X_train
combined_train_X = np.concatenate((X_train, X_test), axis=0)
combined_train_y = np.concatenate((y_train, y_test), axis=0)
print(combined_train_X.shape)
print(combined_train_y.shape)


# In[ ]:


ridge.fit(combined_train_X, combined_train_y)
lasso.fit(combined_train_X, combined_train_y)
elastic=elasticnet.fit(combined_train_X, combined_train_y.ravel())
xgb_model = xgboost.fit(combined_train_X, combined_train_y)
voting_model = vr.fit(combined_train_X, combined_train_y.ravel())
stack_model=stack_reg.fit(combined_train_X, combined_train_y.ravel())


# In[ ]:


final_lasso = np.expm1(lasso.predict(temp_test))
final_ridge = np.expm1(ridge.predict(temp_test))
final_vr = np.expm1(voting_model.predict(temp_test))
final_stck = np.expm1(stack_model.predict(temp_test))


# In[ ]:


final_pred = (0.15*final_vr+0.35*final_lasso+0.15*final_stck+0.35*final_ridge)


# In[ ]:


final_submission = pd.DataFrame({
        "Id": test["Id"],
        "SalePrice": final_pred
    })
final_submission.to_csv("final_submission.csv", index=False)
final_submission.head()


# #### This is not the final notebook, we have not done outlier removal and there is scope of dimensionality reduction.
# However, this is a begginer friendly notebook.
# I shall keep adding more to the notebook.

# In[ ]:




