#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import regression
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV


# In[ ]:


df1=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df2=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df3=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')


# In[ ]:


df1.head()


# In[ ]:


df1.shape


# In[ ]:


df2.head()


# In[ ]:


df2.shape


# In[ ]:


df3.shape


# In[ ]:


df2=pd.concat((df2,df3.iloc[:,-1]),axis=1)


# In[ ]:


df1.shape


# In[ ]:


df2.shape


# In[ ]:


temp=pd.concat((df1,df2),axis=0)


# In[ ]:


temp.shape


# ## Missing Values

# In[ ]:


for i in temp.columns:
    print(i,temp[i].isnull().sum())


# In[ ]:


temp.drop(['Id','Alley','FireplaceQu','Fence','MiscFeature','PoolQC'] ,axis=1,inplace=True)


# In[ ]:


for i in temp.select_dtypes('object').columns :
    m=temp[i].mode()[0]
    temp[i].replace({np.nan : m},inplace=True)


# In[ ]:


temp.info()


# In[ ]:


for i in temp.select_dtypes('float64').columns :
    m=temp[i].median()
    temp[i].replace({np.nan : m},inplace=True)


# In[ ]:


temp.info()


# In[ ]:


df=temp.copy()


# In[ ]:


df.shape


# In[ ]:


for i in df.columns:
    if 'Yr' in i or 'Year' in i and i!='YearBuilt':
        df[i]=df[i]-df['YearBuilt']


# In[ ]:


df.drop(['YearBuilt'],axis=1,inplace=True)


# In[ ]:


for i in df.columns:
    if 'Yr' in i or 'Year' in i:
        print(df[i])


# In[ ]:


plt.scatter(df['YrSold'],df['SalePrice'])


# ### I don't know why heatmap is not working since the docker environment updated?

# In[ ]:


a=df.select_dtypes('object').columns


# In[ ]:


a


# In[ ]:


alldum=pd.DataFrame()


# In[ ]:


for i in a:
    dum=pd.get_dummies(df[i],drop_first=True)
    alldum=pd.concat([alldum,dum],axis=1)
    df=df.drop(i,axis=1)


# In[ ]:


df=pd.concat([alldum,df],axis=1)


# In[ ]:


df.head()


# In[ ]:


df =df.loc[:,~df.columns.duplicated()]


# In[ ]:


df.head()


# In[ ]:


df.describe()


# ## Outliers Handling

# In[ ]:


import seaborn as sns


# In[ ]:


count=0
for i in df.select_dtypes('int64').columns:
    count+=1
print("Integer64 Type count :",count)


# ### since we have encoded and removed object data

# In[ ]:


count=0
for i in df.select_dtypes('object').columns:
    count+=1
print("Object Type count :",count)


# In[ ]:


count=0
for i in df.select_dtypes('float64').columns:
    count+=1
print("Float Type count :",count)


# In[ ]:


for i in df.select_dtypes('int64').columns:
    low=df[i].quantile(.5)
    high=df[i].quantile(.95)
    for j in df[i]:
        if j<low :
            df[i].replace({j:low},inplace=True)
        elif j>high :
            df[i].replace({j:high},inplace=True)


# In[ ]:


for i in df.select_dtypes('float64').columns:
    low=df[i].quantile(.5)
    high=df[i].quantile(.95)
    for j in df[i]:
        if j<low :
            df[i].replace({j:low},inplace=True)
        elif j>high :
            df[i].replace({j:high},inplace=True)


# In[ ]:





# ## Dependent, independent variables

# In[ ]:


X_train=df.iloc[:1459,:-1]
y_train=df.iloc[:1459,-1]
X_test=df.iloc[1460:,:-1]
y_test=df.iloc[1460:,-1]


# In[ ]:


y_test.shape


# # **XGBRegressor**

# In[ ]:


regressor=xgb.XGBRegressor()


# In[ ]:


## Hyper Parameter Optimization


n_estimators = [100, 500, 900, 1100, 1500, 1900]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.03,0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4,5,6,7]
base_score=[0.15,0.25,0.5,0.75,1]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }


# In[ ]:


random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = -1,
            verbose = 5, 
            return_train_score = True,
            random_state=42)


# In[ ]:


random_cv.fit(X_train,y_train)


# In[ ]:


random_cv.best_estimator_


# In[ ]:


regressor=xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.05, max_delta_step=0, max_depth=10,
             min_child_weight=4, missing=np.nan, monotone_constraints='()',
             n_estimators=500, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)


# In[ ]:


regressor.fit(X_train,y_train)


# In[ ]:


y_pred=regressor.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error as mse


# In[ ]:


np.sqrt(mse(np.log(y_test),np.log(y_pred)))


# In[ ]:


y_pred.shape


# In[ ]:


np.sqrt(mse(y_test,y_pred))


# In[ ]:


np.mean(np.array(y_pred))


# ## creating submission.csv to upload

# In[ ]:


sub=pd.DataFrame({'Id': df3.iloc[:,0] , 'SalePrice' : y_pred})


# In[ ]:


sub.head()


# In[ ]:


df3.head()


# In[ ]:


sub.to_csv('Submission.csv',index=False)


# In[ ]:





# In[ ]:





# # **RFRegressor**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1500, random_state = 0, n_jobs=-1)
rf.fit(X_train, y_train)


# In[ ]:


y_pred=rf.predict(X_test)


# In[ ]:


y_pred.mean()


# In[ ]:


mse(y_pred,y_test)


# In[ ]:


np.sqrt(mse(y_test,y_pred))


# ## using lasso model as we have large no of features

# In[ ]:


from sklearn.linear_model import Lasso


# In[ ]:


ls=Lasso()


# In[ ]:


parameters={'alpha' :[1e-15,1e-10,1e-5,1e-4,1e-3,1e-2,.1,1,2,3,5,10,20,50,100,150,200,300,450]}


# In[ ]:


lreg=GridSearchCV(ls,parameters,scoring='neg_mean_squared_error',cv=5)


# In[ ]:


lreg.fit(X,y)


# In[ ]:


print(lreg.best_params_)
print(lreg.best_score_)


# In[ ]:


ls=Lasso(alpha=450)


# In[ ]:


ls.fit(X,y)


# In[ ]:


y_pred=ls.predict(X_test)


# In[ ]:


mse(y_test,y_pred)


# In[ ]:


np.sqrt(mse(y_test,y_pred))


# In[ ]:


np.mean(y_test)


# In[ ]:





# In[ ]:




