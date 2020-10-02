#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder  ###for encode a categorical values
from sklearn.model_selection import train_test_split  ## for spliting the data
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor  
from catboost import CatBoostRegressor
import seaborn as sns


# In[ ]:


train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
train.head()


# In[ ]:


train.isnull().any()


# In[ ]:


miss_col=[col for col in train.columns if train[col].isnull().any()]
print(miss_col)


# In[ ]:


sns.set_style("whitegrid")
missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()


# In[ ]:





# Most of times missing values imply 0, example no garage (GaragCond, GarageType,GarageYrBlt)

# In[ ]:


for col in miss_col:
    train[col]=train[col].fillna(train[col].mode()[0])
    #train[col]=train[col].fillna(train[col].mean())


# In[ ]:


train.isnull().sum()


# In[ ]:





# In[ ]:





# In[ ]:


train.info()


# In[ ]:





# In[ ]:


LE=LabelEncoder()
for col in train.select_dtypes(include=['object']):
    train[col]=LE.fit_transform(train[col])


# In[ ]:


train.head()


# In[ ]:


#standardizing data
saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


# In[ ]:


# Adding total sqfootage feature 
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
print(train)
train.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis=1)


# In[ ]:


test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test.isnull().sum()


# In[ ]:


miss_test=[col for col in test.columns if test[col].isnull().any()]
print(miss_test)


# In[ ]:


for col in miss_test:
    test[col]=test[col].fillna(test[col].mode()[0])


# In[ ]:


test.head()


# In[ ]:


for col in test.select_dtypes(include=['object']):
    test[col]=LE.fit_transform(test[col])

test.head()


# In[ ]:


# Adding total sqfootage feature 
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']
print(test)
test.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis=1)


# In[ ]:


X_train=train.drop(["SalePrice"],axis=1)
Y_train=train["SalePrice"]
print(X_train)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X_train , Y_train ,test_size = 0.3,random_state = 3)


# In[ ]:





# In[ ]:


lightgbm = LGBMRegressor(objective='regression', 
                                      num_leaves=8,
                                      learning_rate=0.0385, 
                                      n_estimators=3500,
                                      max_bin=200, 
                                      bagging_fraction=0.75,
                                      bagging_freq=5, 
                                      bagging_seed=7,
                                      feature_fraction=0.2,
                                      feature_fraction_seed=7,
                                      verbose= 0,
                                      ) 


# In[ ]:


lightgbm1 = LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.01, 
                                       n_estimators=5000,
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       #min_data_in_leaf=2,
                                       #min_sum_hessian_in_leaf=11
                                       )
                                       


# In[ ]:


Catmodel= CatBoostRegressor(iterations=20, learning_rate=1, depth=3, verbose= 0)


# In[ ]:


#from sklearn.linear_model import LogisticRegression

#model = LogisticRegression()
#model.fit(X_train, Y_train)
#model.score(X_test, Y_test)


# In[ ]:


Catmodel.fit(x_train,y_train)


# In[ ]:


cs1 = Catmodel.score(x_train,y_train)
print(cs1)


# In[ ]:


cs2 = Catmodel.score(x_test,y_test)
print(cs2)


# In[ ]:


lightgbm.fit(x_train,y_train)
lightgbm1.fit(x_train,y_train)


# In[ ]:


ls1 = lightgbm.score(x_train,y_train)
print(ls1)


# In[ ]:


lg1 = lightgbm1.score(x_train,y_train)
print(lg1)


# In[ ]:


ls2 = lightgbm.score(x_test,y_test)
print(ls2)


# In[ ]:


lg2 = lightgbm1.score(x_test,y_test)
print(lg2)


# In[ ]:


Catmodel.fit(X_train,Y_train)


# In[ ]:


lightgbm.fit(X_train,Y_train)
lightgbm1.fit(X_train,Y_train)


# In[ ]:



prediction_lightgbm = lightgbm.predict(test)
prediction_lightgbm1 = lightgbm1.predict(test)
print('Ran')


# In[ ]:


prediction_catbooster = Catmodel.predict(test)


# In[ ]:


#submit1=pd.DataFrame()
#submit1['Id']=test['Id']
#submit['SalePrice']=prediction_lightgbm
#submit1['SalePrice']=prediction_catbooster
#submit1.to_csv('submission.csv',index=False)
#print('Ran')


# In[ ]:


submit=pd.DataFrame()
submit['Id']=test['Id']
#submit['SalePrice']=prediction_lightgbm
submit['SalePrice']=prediction_lightgbm1
#submit['SalePrice']=prediction_catbooster
submit.to_csv('submission.csv',index=False)
print('Ran')


# In[ ]:




