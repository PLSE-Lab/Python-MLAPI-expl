#!/usr/bin/env python
# coding: utf-8

# # Data Processing for the House Price Kaggle Competition
# 
# My work has been inspired by the following kernels:
# 
# https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# 
# https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python#3.-Keep-calm-and-work-smart
# 
# https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
# 
# https://www.kaggle.com/apapiu/regularized-linear-models

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[4]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[5]:


#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))


# ## Analyse Target ##

# In[6]:


sns.distplot(train.SalePrice)


# In[7]:


train.SalePrice.describe()


# In[8]:


train.SalePrice.skew()


# In[9]:


train.SalePrice.isna().sum()


# In[10]:


train.SalePrice.isnull().sum()


# SalePrice is skewed to the right, will have to be normalised.
# 
# The maximum seems to be very far from 75% percentile, shows evidence of outliers.
# 
# No 0 value
# 
# No NA or null

# ## Missing Data ##

# In[11]:


ntrain = train.shape[0]
ntest = test.shape[0]
#y_train = train.SalePrice
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# In[13]:


def check_nas():    
    sample_values = pd.DataFrame(index=all_data.columns,columns=['SampleValue'])
    for i in all_data.columns:
        sample_values.loc[i].SampleValue = all_data[i].value_counts().index[1]
    nas = pd.DataFrame(all_data.isnull().sum(),columns=['SumOfNA'])
    types = pd.DataFrame(all_data.dtypes,columns=['Type'])
    sample_values.sort_index(inplace=True)
    nas.sort_index(inplace=True)
    types.sort_index(inplace=True)
    alls=pd.concat([sample_values,nas,types],axis=1)
    return(alls[alls.SumOfNA>0].sort_values('SumOfNA',ascending=False))


# In[14]:


check_nas()


# In[15]:


# Most of the NAs are probably because the property does not contain the specific thing (eg. No Pool)
none_feats = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu',
              'GarageFinish','GarageQual','GarageType','GarageCond',
              'BsmtFinType2','BsmtExposure','BsmtFinType1','BsmtQual',
              'BsmtCond','MasVnrType','MSZoning']
zero_feats = ['LotFrontage','GarageYrBlt','MasVnrArea']


# In[16]:


for i in none_feats:
    all_data[i].fillna('None',inplace=True)


# In[17]:


for i in zero_feats:
    all_data[i].fillna(0,inplace=True)


# In[18]:


all_data.drop(['MasVnrArea','MasVnrType','Electrical'],axis=1,inplace=True)


# In[19]:


check_nas()


# In[20]:


all_data['BsmtFullBath'].fillna(0,inplace=True)
all_data['BsmtHalfBath'].fillna(0,inplace=True)
all_data['Functional'].fillna('Typ',inplace=True)
all_data['Utilities'].fillna('AllPub',inplace=True)
all_data['BsmtFinSF1'].fillna(0,inplace=True)
all_data['BsmtFinSF2'].fillna(0,inplace=True)
all_data['BsmtUnfSF'].fillna(0,inplace=True)
all_data['Exterior1st'].fillna('VinylSd',inplace=True)
all_data['Exterior2nd'].fillna('VinylSd',inplace=True)
all_data['GarageArea'].fillna(0,inplace=True)
all_data['GarageCars'].fillna(0,inplace=True)
all_data['KitchenQual'].fillna('None',inplace=True)
all_data['SaleType'].fillna('WD',inplace=True)
all_data['TotalBsmtSF'].fillna(0,inplace=True)


# In[21]:


check_nas()


# ## Features Importance and Correlation

# In[22]:


fig = plt.figure(figsize=(12,9))
sns.heatmap(train.corr())


# In[23]:


imp_feat=abs(train.corr()['SalePrice']).sort_values(ascending=False).head(11).index


# In[24]:


fig = plt.figure(figsize=(10,5))
sns.heatmap(train[imp_feat].corr(),annot=True)


# GarageCars and GarageArea very correlated --> keep GarageCars
# 
# TotalBsmtSF and 1stFlrSF very correlated --> keep TotalBsmtSF
# 
# GrLivArea and TotRmsAbvGrd very correlated --> keep GrLivArea

# In[25]:


imp_feat_2 = imp_feat.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd'])


# In[26]:


# sub 5
all_data.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd'],axis=1,inplace=True)


# In[27]:


imp_feat_2


# In[28]:


sns.pairplot(train[imp_feat_2])


# ## Outliars

# We first see form the pairplot two outliars in SalePrice vs GrLivArea, let's identify and eliminate those

# In[29]:


fig = plt.figure(figsize=(10,5))
plt.scatter(x=train.GrLivArea,y=train.SalePrice)


# ## Normalising Features and Target

# In[30]:


x=abs(train[imp_feat_2].skew()).sort_values(ascending=False)
x


# In[31]:


def check_skew(column,with_log):
    feat = train[column]
    if (with_log==False):
        fig,ax = plt.subplots(figsize = (5,3))
        ax = sns.distplot(feat,fit=norm)
        fig,ax = plt.subplots(figsize = (5,3))
        ax = stats.probplot(feat, plot=plt)
        (mu, sigma) = norm.fit(feat)
        print( 'The normal dist fit has the following parameters: \n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    elif (with_log==True):
        feat = np.log1p(feat)
        fig,ax = plt.subplots(figsize = (5,3))
        ax = sns.distplot(feat,fit=norm)
        fig,ax = plt.subplots(figsize = (5,3))
        ax = stats.probplot(feat, plot=plt)
        (mu, sigma) = norm.fit(feat)
        print( 'The normal dist fit has the following parameters: \n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    


# In[32]:


check_skew(column='SalePrice',with_log=False)
check_skew(column='SalePrice',with_log=True)


# In[33]:


train['SalePrice'] = np.log1p(train['SalePrice'])


# In[34]:


check_skew(column='TotalBsmtSF',with_log=False)
check_skew(column='TotalBsmtSF',with_log=True)


# Zero values ruin log method, must not include them in log

# In[35]:


all_data['TotalBsmtSF'][all_data['TotalBsmtSF']!=0] =  np.log1p(all_data['TotalBsmtSF'][all_data['TotalBsmtSF']!=0])


# In[36]:


fig,ax = plt.subplots(figsize = (5,3))
ax = sns.distplot(train['TotalBsmtSF'][train['TotalBsmtSF']!=0],fit=norm)
fig,ax = plt.subplots(figsize = (5,3))
ax = stats.probplot(train['TotalBsmtSF'][train['TotalBsmtSF']!=0], plot=plt)
(mu, sigma) = norm.fit(train['TotalBsmtSF'][train['TotalBsmtSF']!=0])
print( 'The normal dist fit has the following parameters: \n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))


# In[37]:


x=abs(train[imp_feat_2].skew()).sort_values(ascending=False)
x


# In[38]:


# for sub 5
#num_nonzero = list(set(all_data.dtypes[all_data.dtypes != "object"].index) & set([i for i in all_data.columns.values if sum(all_data[i]==0)==0]))
#skewed = abs(all_data[num_nonzero].skew()).sort_values(ascending=False)
#skewed_feats = skewed[skewed > 0.75].index
#all_data[skewed_feats] = np.log1p(all_data[skewed_feats])


# In[39]:


# for sub 4
#numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
#numeric_feats
#skewed_feats=abs(all_data[numeric_feats].skew()).sort_values(ascending=False)
#skewed_feats = skewed_feats[skewed_feats > 0.75].index
#all_data[skewed_feats] = np.log1p(all_data[skewed_feats])


# In[40]:


all_data = pd.get_dummies(all_data)


# In[41]:


all_data.shape


# # Principal Component Analysis

# In[42]:


# sub 6
#from sklearn.decomposition import PCA
#pca = PCA(n_components=2)
#pca.fit(all_data)
#x_pca = pca.transform(all_data)
#x_pca = pd.DataFrame(x_pca, columns=['pca1','pca2'])


# In[43]:


#all_data = pd.concat([all_data,x_pca],axis=1)


# # Modelling

# In[44]:


y_train = train.SalePrice
train = all_data[:ntrain]
test = all_data[ntrain:]


# In[45]:


train[train.GrLivArea >= 4600].index


# In[46]:


# drop outliers from train and y_train
# for sub 3
y_train.drop(train[train.GrLivArea >= 4600].index,inplace=True)
train.drop(train[train.GrLivArea >= 4600].index,inplace=True)


# In[ ]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LogisticRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb


# # 1st Round of Models

# In[ ]:


gb = make_pipeline(RobustScaler(), GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5))
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.5, random_state=1,max_iter=5000))
model_xgb = make_pipeline(RobustScaler(), xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))


# # Grid Search 

# In[ ]:


param_grid = {'learning_rate':[0.04,0.05,0.06],'max_depth':[2,4,6]}
search_gb = GridSearchCV(GradientBoostingRegressor(n_estimators=3000,loss='huber', 
                                                   min_samples_leaf=15,
                                                   min_samples_split=10, 
                                                   random_state =5),
                       param_grid = param_grid
                       ,cv=3)
gb_pipe_search = make_pipeline(RobustScaler(),search_gb)
search_gb.fit(X,y)
search_gb.best_params_


# In[ ]:


param_grid = {'learning_rate':[0.04,0.05,0.06],'max_depth':[2,4,6]
             }
search_xgb =GridSearchCV(xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1),param_grid=param_grid,cv=3)
xgb_pipe_search = make_pipeline(RobustScaler(),search_xgb)
search_xgb.fit(X,y)
search_xgb.best_params_


# In[ ]:


param_grid = {'alpha':np.linspace(0.0001,0.01,100),'l1_ratio':[0.3,0.5,0.9]}
search_enet = GridSearchCV(ElasticNet(random_state=3),param_grid=param_grid,cv=3)
enet_pipe_search = make_pipeline(RobustScaler(),search_enet)
search_enet.fit(X,y)
search_enet.best_params_


# # Optimized Models

# In[ ]:


gb_opt = make_pipeline(RobustScaler(), GradientBoostingRegressor(n_estimators=3000, learning_rate=0.04,
                                   max_depth=2, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5))
model_xgb_opt = make_pipeline(RobustScaler(), xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=2, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1))
ENet_opt = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))


# # Averaging Models Class

# In[ ]:


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1) 


# # Fitting Model

# In[ ]:


avg_opt = AveragingModels(models = [gb_opt, model_xgb_opt, ENet_opt])
avg.fit(X,y)


# # Write Submission

# In[ ]:


#sub = pd.DataFrame()
#sub['Id'] = test_ID
#sub['SalePrice'] = np.expm1(avg_opt.predict(test))
#sub.to_csv('submission8.csv',index=False)

