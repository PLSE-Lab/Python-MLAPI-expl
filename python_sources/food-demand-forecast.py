#!/usr/bin/env python
# coding: utf-8

# # Food Demand Forecast(Analytics Vidhya Hackthon)
# Source : https://datahack.analyticsvidhya.com/contest/genpact-machine-learning-hackathon-1/

# # Import 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


# In[ ]:


from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.model_selection import GridSearchCV


# # Read

# In[ ]:


dftrain = pd.read_csv('../input/train.csv')
print(dftrain.shape)
dftrain.head()


# In[ ]:


dfmeal = pd.read_csv('../input/meal_info.csv')
print(dfmeal.shape)
dfmeal.head()


# In[ ]:


dfcenter = pd.read_csv('../input/fulfilment_center_info.csv')
print(dfcenter.shape)
dfcenter.head()


# # Merged Df

# In[ ]:


food = pd.merge(pd.merge(dftrain,dfmeal,on='meal_id'),dfcenter,on='center_id')
food.head()


# In[ ]:


dftest = pd.read_csv('../input/test_QoiMO9B.csv')
print(dftest.shape)
dftest.head()


# In[ ]:


foodTest = pd.merge(pd.merge(dftest,dfmeal,on='meal_id'),dfcenter,on='center_id')
foodTest.head()


# In[ ]:


food.info()


# In[ ]:


foodTest.info()


# # Feature Eng

# In[ ]:


food.loc[food[food['checkout_price'] > food['base_price']].index,'Discount'] = 1
food.loc[food[food['checkout_price'] < food['base_price']].index,'Discount'] = -1
food.loc[food[food['checkout_price'] == food['base_price']].index,'Discount'] = 0


# In[ ]:


#food['promotion'] = food['emailer_for_promotion'] + food['homepage_featured']


# In[ ]:


#food['cuisine'] = food['cuisine'].replace(['Thai','Indian','Italian'],['Asian','Asian','Continental'])


# In[ ]:


food['year'] = food['week'].apply(lambda x: int(x/52))
food['month'] = food['week'].apply(lambda x: int(x/4))


# In[ ]:


food.drop(['checkout_price','city_code','region_code'],axis=1,inplace=True)


# In[ ]:


food.head()


# In[ ]:


foodTest.loc[foodTest[foodTest['checkout_price'] > foodTest['base_price']].index,'Discount'] = 1
foodTest.loc[foodTest[foodTest['checkout_price'] < foodTest['base_price']].index,'Discount'] = -1
foodTest.loc[foodTest[foodTest['checkout_price'] == foodTest['base_price']].index,'Discount'] = 0


# In[ ]:


#foodTest['promotion'] = foodTest['emailer_for_promotion'] + foodTest['homepage_featured']
#foodTest['cuisine'] = foodTest['cuisine'].replace(['Thai','Indian','Italian','Continental''],['Asian','Asian','Continental'])
foodTest['year'] = foodTest['week'].apply(lambda x: int(x/52))
foodTest['month'] = foodTest['week'].apply(lambda x: int(x/4))
foodTest.drop(['checkout_price','city_code','region_code'],axis=1,inplace=True)


# # OH Encode

# In[ ]:


dummyTrain = pd.get_dummies(food.drop(['id','op_area'],axis=1))
dummyTrain.head()


# In[ ]:


dummyTest = pd.get_dummies(foodTest.drop(['id','op_area'],axis=1))


# In[ ]:


sc = StandardScaler()


# # Scaling

# In[ ]:


scaledTrain = pd.DataFrame(sc.fit_transform(dummyTrain),columns=dummyTrain.columns)
scaledTrain.head()


# In[ ]:


scaledTest = pd.DataFrame(sc.fit_transform(dummyTest),columns=dummyTest.columns)
scaledTest.head()


# In[ ]:


x = scaledTrain.drop('num_orders',axis=1)
y = scaledTrain['num_orders']


# # Linear Regression

# In[ ]:


lr = LinearRegression()


# In[ ]:


lr.fit(x,y)


# In[ ]:


ypredLr = lr.predict(scaledTest)
ypredLr


# # Inverse Transformation of Target

# In[ ]:


sci = StandardScaler()


# In[ ]:


sci.fit_transform(pd.DataFrame(food['num_orders']))


# In[ ]:


ypredlr_i = sci.inverse_transform(ypredLr)
ypredlr_i


# In[ ]:


ypredlr_i = abs(ypredlr_i)


# In[ ]:


pd.DataFrame({'id':foodTest['id'],'num_orders':ypredlr_i})


# In[ ]:


params = {"alpha":[0.01,0.5,1,2,3,4,0.02,0.03,0.09,10,50],
         "solver":["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
         "random_state":[0,2,1,3,500]}
params_lasso = {"alpha":[0.01,0.5,1,2,3,0.05,0.02,0.03,0.09,0.001,5],
         "random_state":[0,2,1,3,500]}
params_elastic = {"alpha":[0.01,0.5,1,2,3,0.05,0.02,0.03,0.001,5],
         "random_state":[0,2,1,3,500]}


# # Lasso

# In[ ]:


lasso = Lasso(alpha=0.01,random_state=0)
lasso.fit(x,y)
ypredl = lasso.predict(scaledTest)
ypredl_i = sci.inverse_transform(ypredl)
ypredl_i


# In[ ]:


ypredl_i = abs(ypredl_i)


# In[ ]:


pd.DataFrame({'id':foodTest['id'],'num_orders':ypredl_i})


# In[ ]:


gLasso =GridSearchCV(estimator=lasso,param_grid=params_lasso,cv=3)


# In[ ]:


#gLasso.fit(x,y)


# In[ ]:


#gLasso.best_params_


# In[ ]:


plt.figure(figsize=(12,12))
pd.Series(lasso.coef_,index=x.columns).plot(kind='barh')


# # ElasticNet

# In[ ]:


enet = ElasticNet(alpha=0.01,random_state=0)
enet.fit(x,y)
ypredenet = enet.predict(scaledTest)
ypredenet_i = sci.inverse_transform(ypredenet)
ypredenet_i


# In[ ]:


ypredenet_i = abs(ypredenet_i)


# In[ ]:


pd.DataFrame({'id':foodTest['id'],'num_orders':ypredenet_i})


# In[ ]:


gEnet =GridSearchCV(estimator=enet,param_grid=params_elastic,cv=3)


# In[ ]:


#gEnet.fit(x,y)


# In[ ]:


#gEnet.best_params_


# # Heatmap

# In[ ]:


plt.figure(figsize=(12,12))
sns.heatmap(food.corr(),annot=True)


# Final score on analytics vidhya after submission,
# RMSE : 99.2
