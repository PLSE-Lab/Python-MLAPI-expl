#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import common libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy.stats import skew, skewtest, norm


# In[ ]:


train =pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


test.shape


# In[ ]:


#Lets take a quick look at distribution of target variable-SalePrice
fig,ax = plt.subplots(figsize=(8,5))
sns.distplot(train['SalePrice'],fit=norm,ax=ax)


# It is evident from above figure that skew is present in SalePrice.

# In[ ]:


print("Skew is: %f" % train['SalePrice'].skew())


# In[ ]:


#log transform the target
train['SalePrice'] = np.log1p(train['SalePrice'])


# In[ ]:


train['SaleCondition'].value_counts()


# In[ ]:


#Let's look at SalePrice vs living area
plt.scatter(train['GrLivArea'],train['SalePrice'],color='blue')
plt.ylabel('SalePrice')
plt.xlabel('Living Area')


# We can see from above plot that some outliers are present as for some large living area, 
# the price is very low.

# In[ ]:


#let's remove outliers
train = train[train['GrLivArea']<4500]


# In[ ]:


#function to find average price by feature passed as input
def avgprice(grpby) :
    avgprc = pd.DataFrame(train['SalePrice'].groupby(train[grpby]).agg('mean'))
    return avgprc


# In[ ]:


#let's check average saleprice with respect to neighborhood
#avgprice = pd.DataFrame(train['SalePrice'].groupby(train['Neighborhood']).agg('mean'))
#type(avgprice)
fig,ax = plt.subplots(figsize=(12,5))
plt.setp(ax.get_xticklabels(), rotation=45)
sns.barplot(x=avgprice('Neighborhood').index,y=avgprice('Neighborhood')['SalePrice'],ax=ax)


# Above plot shows that house prices in Northridge, Northridge Heights & Stone Brook area are
# comparitively higher than other areas.

# In[ ]:


#Let's explore the saleprice of house with yearbuilt
#avgprice_conf = pd.DataFrame(train['SalePrice'].groupby(train['LotConfig']).agg('mean'))
sns.barplot(x=avgprice('LotConfig').index,y=avgprice('LotConfig')['SalePrice'])


# The average SalePrice of houses with lotconfig Cul-De-Sac(closed end street) and 
# FR3(3 side road facing) are significantly higher than others.

# In[ ]:


#lets find the impact of building type on average saleprice
sns.barplot(x=avgprice('BldgType').index,y=avgprice('BldgType')['SalePrice'])


# SalePrice of building type 1Family detached and Townhouse Inside unit is higher than others.

# In[ ]:


#lets explore the relationship between zoning classification(MSZoning) and SalePrice
sns.barplot(x=avgprice('MSZoning').index,y=avgprice('MSZoning')['SalePrice'])


# We can see that low density residential areas and floating village residential areas have higher
# average saleprice as compared to other zoning classifications.

# In[ ]:


sns.distplot(train['YearBuilt'])


# Here we can see that number of houses built started increasing from around 1950 till 2005-06 and 
# after that it started decreasing sharply after 2008 probably due to worldwide economic crisis.

# In[ ]:





# ## Handle Missing Values

# In[ ]:


combined = pd.concat((train.loc[:,'MSSubClass' :'SaleCondition'],test.loc[:,'MSSubClass' :'SaleCondition']))


# In[ ]:


#Pick the first one - LotFrontage
combined['LotFrontage'].value_counts()


# In[ ]:


plt.scatter(combined['LotFrontage'],combined['LotArea'])


# In[ ]:


fig,ax = plt.subplots(figsize=(12,8))
sns.heatmap(combined.corr(),ax=ax)


# In[ ]:


combined = pd.get_dummies(combined)
combined = combined.fillna(combined.mean())


# In[ ]:


X_train = combined[:train.shape[0]]
X_test = combined[train.shape[0]:]
y_train=train['SalePrice']


# In[ ]:


X_train.head()


# In[ ]:


from sklearn.linear_model import LinearRegression,Lasso,Ridge,SGDRegressor,ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold,cross_val_predict,cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import xgboost as xgb
import lightgbm as lgb


# In[ ]:


n_folds = 5
def rmse_cv(model):
    #kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 10))
    return(rmse)


# In[ ]:


#linear regression
lr =LinearRegression()
# scores = cross_val_score(lr, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
# rmse_scores = np.sqrt(-scores)
print(rmse_cv(lr).mean())


# In[ ]:


#Ridge regression
rr = Ridge(alpha=0.2,normalize=True)
# rr.fit(X_train,y_train)
# scores = cross_val_score(rr, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
# rmse_scores = np.sqrt(-scores)
print(rmse_cv(rr).mean())


# In[ ]:


#Lasso regression
lsr = Lasso(alpha=0.001)
# lsr.fit(X_train,y_train)
# scores = cross_val_score(lsr, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
# rmse_scores = np.sqrt(-scores)
print(rmse_cv(lsr).mean())


# In[ ]:


# Seperate out numeric and categorical feature standard scaling
cols = train.loc[:,'MSSubClass' :'SaleCondition'].columns
cols_scale = []
# def str_column_to_float(dataset, column):
#        for row in dataset:
#           if (row[column] = float(row[column].strip())
for col in cols :
    if train[col].dtypes != 'O' :
        cols_scale.append(col)

cols_noscale = list(set(X_train.columns).symmetric_difference(set(cols_scale)))
#X_train['SaleType_New'].unique()
#type([0,1])
#train['PoolQC'].dtypes


# In[ ]:


#SGD regression
sgd = SGDRegressor(random_state=0,max_iter=300,alpha=0.02,penalty='elasticnet',l1_ratio=0.1,
                   power_t=0.4)
#X_train1 = pd.DataFrame(StandardScaler().fit_transform(X_train),columns=list(X_train.columns))
X_train1 = pd.DataFrame(StandardScaler().fit_transform(X_train[cols_scale]),columns=cols_scale)
X_train2 = pd.concat([X_train1.reset_index(drop=True),X_train[cols_noscale].reset_index(drop=True)],axis=1)

#y_train1 = StandardScaler().fit_transform(y_train)
sgd.fit(X_train1,y_train)
scores = cross_val_score(sgd,X_train1,y_train,scoring='neg_mean_squared_error',cv=10)
rmse_scores = np.sqrt(-scores)
print(rmse_scores.mean())


# In[ ]:


#GradientBoosting Regressor
gdb = GradientBoostingRegressor(n_estimators=400,max_features='sqrt',alpha=0.9)
# gdb.fit(X_train,y_train)
# scores = cross_val_score(gdb, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
# rmse_scores = np.sqrt(-scores)
print(rmse_cv(gdb).mean())


# In[ ]:


#RandomForest regressor
rfr = RandomForestRegressor(n_estimators=100)
# rfr.fit(X_train,y_train)
# scores = cross_val_score(rfr, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
# rmse_scores = np.sqrt(-scores)
print(rmse_cv(rfr).mean())


# In[ ]:


#ElasticNet Regressor
elnr = ElasticNet(alpha=0.001,l1_ratio=0.3,max_iter=3000)
# elnr.fit(X_train,y_train)
# scores = cross_val_score(elnr, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
# rmse_scores=np.sqrt(-scores)
print(rmse_cv(elnr).mean())


# In[ ]:


#Kernel-ridge
krr = KernelRidge(alpha=0.6,degree=2,kernel='polynomial',coef0=2.7)
krr.fit(X_train1,y_train)
scores = cross_val_score(krr, X_train1, y_train, scoring="neg_mean_squared_error", cv=10)
rmse_scores=np.sqrt(-scores)
print(rmse_scores.mean())


# In[ ]:


#Xgb regressor
xgbr = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
# xgbr.fit(X_train,y_train)
# scores = cross_val_score(xgbr, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
# rmse_scores=np.sqrt(-scores)
print(rmse_cv(xgbr).mean())


# In[ ]:


#lightgbm regressor
lgbr = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=700,
                              max_bin = 55, bagging_fraction = 0.4,
                              bagging_freq = 5, feature_fraction = 0.23,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
# lgbr.fit(X_train,y_train)
# scores = cross_val_score(lgbr, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
# rmse_scores=np.sqrt(-scores)
print(rmse_cv(lgbr).mean())


# In[ ]:


#function to take average of lasso, LightGBM ,XGB and ElasticNet
def averaging_model(model1,model2,model3,model4):
    model1.fit(X_train,y_train)
    model2.fit(X_train,y_train)
    model3.fit(X_train,y_train)
    model4.fit(X_train,y_train)
    pred1 = model1.predict(X_test)
    pred2 = model2.predict(X_test)
    pred3 = model3.predict(X_test)
    pred4 = model4.predict(X_test)
    prediction = pd.DataFrame()
    prediction['pred1'] = pred1
    prediction['pred2'] = pred2
    prediction['pred3'] = pred3
    prediction['pred4'] = pred4
    prediction = np.mean(prediction,axis=1)
    return prediction


# In[ ]:


pred = averaging_model(lsr,lgbr,xgbr,elnr)


# In[ ]:


test['SalePrice'] =np.expm1(pred) #convert back from log to normal SalePrice
output = test[['Id','SalePrice']]
output.to_csv('output.csv',index=False)


# In[ ]:





# In[ ]:




