#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 2000)


# In[ ]:


h_train=pd.read_csv("../input/house-prices-advanced-regression-techniques")
h_train.head()


# In[ ]:


h_train.dtypes.head()


# In[ ]:


h_train.isnull().sum().head()


# In[ ]:


total_missing=h_train.isnull().sum().sort_values()
percMissing = h_train.isnull().sum() / h_train.isnull().count().sort_values()*100
missing = pd.concat([total_missing, percMissing], axis = 1, keys = ['total #', '%'])
missing[missing['total #'] > 0]


# In[ ]:


## as we can see there are 4 features having  more than 80% null value it's better to drop that features rather than try to fill them

h_train.drop(["PoolQC","MiscFeature","Fence","Alley"],axis=1,inplace=True)


# ### analysing 'SalePrice

# In[ ]:


h_train['SalePrice'].describe()


# In[ ]:


sns.distplot(h_train['SalePrice']);


# #### We have positive skewness.

# In[ ]:


#skewness and kurtosis
print("Skewness: %f" % h_train['SalePrice'].skew())
print("Kurtosis: %f" % h_train['SalePrice'].kurt())


# In[ ]:


#scatter plot GrLivArea/saleprice
var = 'GrLivArea'
data = pd.concat([h_train['SalePrice'], h_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# #### linear relationship

# In[ ]:


#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([h_train['SalePrice'], h_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# #### linear but not as above

# In[ ]:


# Relationship with categorical features
sns.barplot(h_train.OverallQual,h_train.SalePrice)


# #### As we can see that higher the Quality higher the price

# In[ ]:


plt.subplots(figsize=(12, 9))
sns.heatmap(h_train.corr())


# In[ ]:


#'SalePrice' correlation matrix (zoomed heatmap style) take only those columns from upper heatmap
col=h_train[['SalePrice','GarageYrBlt','OverallQual','GarageCars','GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF','YearBuilt','TotRmsAbvGrd']]
col.corr()


# In[ ]:


h_train.shape


# In[ ]:


print("Find most important features relative to target")
corr = h_train.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
print(corr.SalePrice)


# In[ ]:


sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(h_train[cols], size = 2.5)
plt.show();


# ###  These all the columns which have null values
# ###  Take one coloumn at a time for missing values

# In[ ]:


h_train[['FireplaceQu','LotFrontage','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'BsmtQual','Electrical','GarageCond','GarageFinish','GarageQual','GarageType','GarageYrBlt','MasVnrArea','MasVnrType']].dtypes


# In[ ]:


h_train.shape


# ### As we can see only three variable are float so we can check their corelation with SalePrice

# In[ ]:


h_train[['LotFrontage','SalePrice']].corr()


# In[ ]:


sns.scatterplot(x = 'SalePrice', y = 'LotFrontage', data = h_train)


# In[ ]:


h_train[['GarageYrBlt','SalePrice']].corr()


# In[ ]:


sns.scatterplot(x = 'SalePrice', y = 'GarageYrBlt', data = h_train)


# In[ ]:


h_train[['MasVnrArea','SalePrice']].corr()


# In[ ]:


sns.scatterplot(x = 'SalePrice', y = 'MasVnrArea', data = h_train)


# ### As we can see there is a correlation with SalePrice so we can not simply delete null values so we can replace null with median so spread will not change

# In[ ]:


h_train['LotFrontage'].replace(np.nan,h_train.LotFrontage.mean(),inplace=True)


# In[ ]:


h_train['GarageYrBlt'].replace(np.nan,h_train.GarageYrBlt.mean(),inplace=True)


# In[ ]:


h_train['MasVnrArea'].replace(np.nan,h_train.MasVnrArea.mean(),inplace=True)


# In[ ]:


h_train.isnull().sum()


# In[ ]:


h_train.drop(h_train.loc[h_train['Electrical'].isnull()].index,inplace=True)


# In[ ]:


h_train['Electrical'].isnull().sum()


# In[ ]:


h_train.shape


# #### Now we handle numerical values so its time to fill categorical values

# In[ ]:


# h_train['Alley'].unique()


# In[ ]:


# h_train['Alley'].replace(np.nan,'No_alley_access',inplace=True)

# sns.countplot(data=h_train,x='Alley')

# #nan replaced with No alley access as per the Data Dictionary


# In[ ]:


#BsmtQual
h_train['BsmtQual'].unique()


# as per the Data Dictionary nan stands for "No Basement"
#so,
h_train['BsmtQual'].replace(np.nan,'No_Basement',inplace=True)


# In[ ]:


sns.countplot(data=h_train,x='BsmtQual')


# In[ ]:


#BsmtCond
h_train['BsmtCond'].unique()

# as per the Data Dictionary nan stands for "No Basement"
#so,
h_train['BsmtCond'].replace(np.nan,'No_Basement',inplace=True)

sns.countplot(data=h_train,x='BsmtCond')


# In[ ]:


#BsmtExposure
h_train['BsmtExposure'].unique()

# as per the Data Dictionary nan stands for "No Basement"
#so,
h_train['BsmtExposure'].replace(np.nan,'No_Basement',inplace=True)

sns.countplot(data=h_train,x='BsmtExposure')


# In[ ]:


#BsmtFinType1
h_train['BsmtFinType1'].unique()

# as per the Data Dictionary nan stands for "No Basement"
#so,
h_train['BsmtFinType1'].replace(np.nan,'No_Basement',inplace=True)

sns.countplot(data=h_train,x='BsmtFinType1')


# In[ ]:


#BsmtFinType2
h_train['BsmtFinType2'].unique()

# as per the Data Dictionary nan stands for "No Basement"
#so,
h_train['BsmtFinType2'].replace(np.nan,'No_Basement',inplace=True)

sns.countplot(data=h_train,x='BsmtFinType2')


# In[ ]:


#FireplaceQu
h_train['FireplaceQu'].unique()

# as per the Data Dictionary nan stands for "No Fireplace"
#so,
h_train['FireplaceQu'].replace(np.nan,'No_Fireplace',inplace=True)

sns.countplot(data=h_train,x='FireplaceQu')


# In[ ]:


#GarageType
h_train['GarageType'].unique()

# as per the Data Dictionary nan stands for "No Garage"
#so,
h_train['GarageType'].replace(np.nan,'No_Garage',inplace=True)

sns.countplot(data=h_train,x='GarageType')


# In[ ]:


#GarageFinish
h_train['GarageFinish'].unique()

# as per the Data Dictionary nan stands for "No Garage"
#so,
h_train['GarageFinish'].replace(np.nan,'No_Garage',inplace=True)

sns.countplot(data=h_train,x='GarageFinish')


# In[ ]:


#GarageQual
h_train['GarageQual'].unique()

# as per the Data Dictionary nan stands for "No Garage"
#so,
h_train['GarageQual'].replace(np.nan,'No_Garage',inplace=True)

sns.countplot(data=h_train,x='GarageQual')


# In[ ]:


#GarageCond
h_train['GarageCond'].unique()

# as per the Data Dictionary nan stands for "No Garage"
#so,
h_train['GarageCond'].replace(np.nan,'No_Garage',inplace=True)

sns.countplot(data=h_train,x='GarageCond')


# In[ ]:


# #PoolQC
# h_train['PoolQC'].unique()

# # as per the Data Dictionary nan stands for "No Pool"
# #so,
# h_train['PoolQC'].replace(np.nan,'No_Pool',inplace=True)

# sns.countplot(data=h_train,x='PoolQC')


# In[ ]:


# #Fence
# h_train['Fence'].unique()

# # as per the Data Dictionary nan stands for "No Fence"
# #so,
# h_train['Fence'].replace(np.nan,'No_Fence',inplace=True)

# sns.countplot(data=h_train,x='Fence')


# In[ ]:


# #MiscFeature
# h_train['MiscFeature'].unique()

# # as per the Data Dictionary nan stands for "None"
# #so,
# h_train['MiscFeature'].replace(np.nan,'None',inplace=True)

# sns.countplot(data=h_train,x='MiscFeature')


# In[ ]:


h_train.shape


# In[ ]:


#MasVnrType
h_train['MasVnrType'].unique()

#in this there is no designation for nan so we are removing the nan values
h_train.drop(h_train.loc[h_train['MasVnrType'].isnull()].index,inplace=True)


# In[ ]:


sns.heatmap(h_train.isnull())


# ##### Now we are clear with the Nan values present in Dataset

# ### Now its for outliers

# In[ ]:


# #standardizing data
# saleprice_scaled = StandardScaler().fit_transform(h_train['SalePrice'][:,np.newaxis]);
# low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
# high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
# print('outer range (low) of the distribution:')
# print(low_range)
# print('\nouter range (high) of the distribution:')
# print(high_range)


# In[ ]:


# sns.distplot(h_train['SalePrice'], fit=norm);
# fig = plt.figure()
# res = stats.probplot(h_train['SalePrice'], plot=plt)


# In[ ]:


# h_train['SalePrice'].quantile([0.1,0.2,0.3,0.4])
# ### as we can see from above graph there is one outlier at -3 std and 2 at +3 std and same we can see below also


# In[ ]:


# h_train['SalePrice'].quantile([0.97,0.98,0.99,1])


# Ok, 'SalePrice' is not normal. It shows 'peakedness', positive skewness and does not follow the diagonal line.
# 
# But everything's not lost. A simple data transformation can solve the problem. This is one of the awesome things you can learn in statistical books: in case of positive skewness, log transformations usually works well. When I discovered this, I felt like an Hogwarts' student discovering a new cool spell.
# 

# In[ ]:


# #applying log transformation
# h_train['SalePrice'] = np.log(h_train['SalePrice'])


# In[ ]:


# #transformed histogram and normal probability plot
# sns.distplot(h_train['SalePrice'], fit=norm);
# fig = plt.figure()
# res = stats.probplot(h_train['SalePrice'], plot=plt)


# In[ ]:


# h_train['SalePrice'].quantile([0.1,0.2,0.3,0.4])


# In[ ]:


# h_train['SalePrice'].quantile([0.97,0.98,0.99,1])


# In[ ]:


# h_train.drop(h_train[h_train['SalePrice']<11.728037].index,axis=0,inplace=True)


# In[ ]:


# h_train.drop(h_train[h_train['SalePrice']>12.993142].index,axis=0,inplace=True)


# In[ ]:


# h_train=h_train.drop('Id',axis=1)


# In[ ]:


# h_train.shape


# In[ ]:


# #LotFrontage
# sns.distplot(h_train['LotFrontage'], fit=norm);
# fig = plt.figure()
# res = stats.probplot(h_train['LotFrontage'], plot=plt)


# In[ ]:


# h_train['LotFrontage'].quantile([0.1,0.2,0.3,0.4])


# In[ ]:


# h_train['LotFrontage'].quantile([0.96,0.97,0.98,0.99,1])


# In[ ]:


# h_train.drop(h_train[h_train['LotFrontage']>139.2].index,axis=0,inplace=True)


# In[ ]:


# h_train.shape


# In[ ]:


# #LotArea
# sns.distplot(h_train['GrLivArea'], fit=norm);
# fig = plt.figure()
# res = stats.probplot(h_train['GrLivArea'], plot=plt)


# In[ ]:


# h_train['GrLivArea'].quantile([0.1,0.2,0.3,0.4])


# In[ ]:


# h_train['GrLivArea'].quantile([0.97,0.98,0.99,1])


# In[ ]:


# h_train.drop(h_train[h_train['GrLivArea']>2931.84].index,axis=0,inplace=True)


# In[ ]:


# h_train.shape


# In[ ]:


# sns.distplot(h_train['TotalBsmtSF'], fit=norm);
# fig = plt.figure()
# res = stats.probplot(h_train['TotalBsmtSF'], plot=plt)


# In[ ]:


# h_train['TotalBsmtSF'].quantile([0.1,0.2,0.3,0.4])


# In[ ]:


# h_train['TotalBsmtSF'].quantile([0.97,0.98,0.99,1])


# In[ ]:


# h_train.drop(h_train[h_train['TotalBsmtSF']<814.0].index,axis=0,inplace=True)


# In[ ]:


# h_train.drop(h_train[h_train['TotalBsmtSF']>2077.84].index,axis=0,inplace=True)


# In[ ]:


# h_train.shape


# ## Now its for model building

# In[ ]:


h_train.drop('Id',axis=1,inplace=True)


# In[ ]:


h_train.corr()


# In[ ]:


h1_train=h_train[["SalePrice","OverallQual","YearBuilt","TotalBsmtSF","1stFlrSF","GrLivArea","FullBath","GarageCars","GarageArea","TotRmsAbvGrd"]]


# In[ ]:


# These are the best correlation with saleprice
# OverallQual      0.790982
# GrLivArea        0.708624
# GarageCars       0.640409
# GarageArea       0.623431
# TotalBsmtSF      0.613581
# 1stFlrSF         0.605852
# FullBath         0.560664
# TotRmsAbvGrd     0.533723
# YearBuilt        0.522897


# In[ ]:


h1_train.shape


# In[ ]:


h1_train['TotRmsAbvGrd'].dtype


# In[ ]:


h1_train_dum=pd.get_dummies(h1_train,drop_first=True)


# In[ ]:


h1_train_dum.shape


# In[ ]:





# #### As we can see TotRmsAbvGrd,1stFlrSF and GarageYrBlt are less correlated

# In[ ]:


x=h1_train_dum.drop(['SalePrice'],axis=1)
y=h1_train_dum['SalePrice']


# In[ ]:


x.shape


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x, y , test_size=0.2 , random_state=21 )


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# In[ ]:


clf = RandomForestRegressor()

param_dist = {"n_estimators": [50, 100, 150,200]}

clf.fit(x_train, y_train)


# In[ ]:


y_pred=clf.predict(x_test)


# In[ ]:


print('MAE:',metrics.mean_absolute_error(y_test,y_pred))

print('*'*20)


print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print('*'*20)


r2_score=metrics.r2_score(y_test,y_pred)
print('r2_score:',r2_score)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


clf_lr=LinearRegression()

clf_lr.fit(x_train,y_train)


# In[ ]:


y_pred_lr=clf_lr.predict(x_test)


# In[ ]:


print('MAE:',metrics.mean_absolute_error(y_test,y_pred_lr))

print('*'*20)


print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred_lr)))
print('*'*20)


r2_score=metrics.r2_score(y_test,y_pred_lr)
print('r2_score:',r2_score)


# ## Cross validation

# In[ ]:


from sklearn.model_selection import cross_val_score,KFold


# In[ ]:


kf=KFold(n_splits=5)
RFRegressor=RandomForestRegressor(random_state=5)

score=cross_val_score(RFRegressor,x,y,cv=kf,scoring='neg_mean_squared_error')

r=score.mean()
print(r)


# In[ ]:


from math import sqrt

sqrt(-r)


# In[ ]:


## use this
kf=KFold(n_splits=5)
LRegressor=LinearRegression()

score=cross_val_score(LRegressor,x,y,cv=kf,scoring='neg_mean_squared_error')

r=score.mean()
print(r)


# In[ ]:


from math import sqrt

sqrt(-r)


# In[ ]:


import xgboost as xgb


# In[ ]:


model = xgb.XGBRegressor()

model.fit(x_train,y_train)


# In[ ]:


y_pred_xgb=model.predict(x_test)

y_pred_xgb


# In[ ]:


print('MAE:',metrics.mean_absolute_error(y_test,y_pred_xgb))

print('*'*20)


print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred_xgb)))
print('*'*20)


r2_score=metrics.r2_score(y_test,y_pred_xgb)
print('r2_score:',r2_score)


# ## XGB with CV

# In[ ]:


kf=KFold(n_splits=5)
xgbRegressor=xgb.XGBRegressor()

score=cross_val_score(xgbRegressor,x,y,cv=kf,scoring='neg_mean_squared_error')

r=score.mean()
print(r)


# In[ ]:


from math import sqrt

sqrt(-r)


# # Test data

# In[ ]:


h_test=pd.read_csv("test.csv")
h_test.head()


# In[ ]:


h_test.shape


# In[ ]:


total_missing_t=h_test.isnull().sum().sort_values()
percMissing_t = h_test.isnull().sum() / h_test.isnull().count().sort_values()*100
missing_t = pd.concat([total_missing_t, percMissing_t], axis = 1, keys = ['total #', '%'])
missing_t[missing_t['total #'] > 0]


# In[ ]:


# h_test['Alley'].replace(np.nan,'No_alley_access',inplace=True)

# #sns.countplot(data=h_train,x='Alley')


# In[ ]:


#BsmtCond
# h_test['BsmtCond'].replace(np.nan,'No_Basement',inplace=True)


# In[ ]:


#BsmtExposure
# h_test['BsmtExposure'].replace(np.nan,'No_Basement',inplace=True)


# In[ ]:


h1_test=h_test[["OverallQual","YearBuilt","TotalBsmtSF","1stFlrSF","GrLivArea","FullBath","GarageCars","GarageArea","TotRmsAbvGrd"]]


# In[ ]:


h1_test.isnull().sum()


# In[ ]:


#GarageCars
h1_test.dtypes
# h_test['GarageCars'].replace(np.nan,'No_Basement',inplace=True)


# In[ ]:


h1_test.drop(h1_test.loc[h1_test['GarageCars'].isnull()].index,inplace=True)


# In[ ]:


h1_test.drop(h1_test.loc[h1_test['TotalBsmtSF'].isnull()].index,inplace=True)


# In[ ]:


h1_test_dum=pd.get_dummies(h1_test,drop_first= True)


# In[ ]:


y_pred_xgb_test=model.predict(h1_test_dum)


# In[ ]:


y_pred_xgb_test=pd.DataFrame(y_pred_xgb_test)


# In[ ]:


y_pred_xgb_test.head()


# In[ ]:


sample=pd.read_csv('sample_submission.csv')


# In[ ]:


sample.head()


# In[ ]:


submit=pd.concat([sample.Id,y_pred_xgb_test],axis=1)


# In[ ]:


submit.head()


# In[ ]:


submit.columns=["Id","SalePrice"]


# In[ ]:


# sns.lmplot("Id","SalePrice",data=submit,fit_reg=True)


# In[ ]:


submit.to_csv("Submission_HLP_kaggle.csv",index=False)


# In[ ]:


submit.shape


# In[ ]:


submit.loc[submit["SalePrice"].isnull()]


# # Hyperparameter Tuning

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from numpy import nan


# In[ ]:


import xgboost as xgb
model = xgb.XGBRegressor()

model.fit(x_train,y_train)


# In[ ]:


Booster=["gbtree","gblinear"]
base_score=[0.25,0.50,0.75,1]


# In[ ]:


n_estimators=[100,500,900,1000,1500]
max_depth=[2,3,5,10,15]
Booster=["gbtree","gblinear"]
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]

hyperparameter_grid={
    "n_estimators":n_estimators,
    "max_depth":max_depth,
    "Booster":Booster,
    "learning_rate":learning_rate,
    "min_child_weight":min_child_weight,
    "base_score":base_score
    
}


# In[ ]:


random_cv=RandomizedSearchCV(estimator=model,
                            param_distributions=hyperparameter_grid,
                            cv=5,n_iter=50,
                            scoring="neg_mean_absolute_error",n_jobs=4,
                            verbose=5,
                            return_train_score=True,
                            random_state=42)


# In[ ]:


random_cv.fit(x_train,y_train)


# In[ ]:


random_cv.best_estimator_


# In[ ]:


regressor=xgb.XGBRegressor(Booster='gbtree', base_score=0.5, booster=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
             gamma=0, gpu_id=-1, importance_type='gain',
             interaction_constraints=None, learning_rate=0.15, max_delta_step=0,
             max_depth=2, min_child_weight=2, missing=nan,
             monotone_constraints=None, n_estimators=100, n_jobs=0,
             num_parallel_tree=1, objective='reg:squarederror', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method=None, validate_parameters=False, verbosity=None)


# In[ ]:


regressor.fit(x_train,y_train)


# In[ ]:


y_pred_ran=regressor.predict(h1_test_dum)


# In[ ]:


y_pred_ran


# In[ ]:


y_pred_ran=pd.DataFrame(y_pred_ran)


# In[ ]:


submit_ran=pd.concat([sample.Id,y_pred_ran],axis=1)


# In[ ]:


submit_ran.head()


# In[ ]:


submit_ran.columns=["Id","SalePrice"]


# In[ ]:


submit_ran.to_csv("Submission_HLP_ran_kaggle.csv",index=False)


# In[ ]:




