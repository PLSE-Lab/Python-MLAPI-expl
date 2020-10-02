#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


import scipy.stats as stats
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import r2_score,mean_squared_error


# In[ ]:


airbnb = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
airbnb.head()


# In[ ]:


airbnb.info()


# In[ ]:


airbnb.isnull().sum()


# In[ ]:


airbnb.drop(['name','host_name'],axis=1,inplace=True)


# In[ ]:


airbnb[airbnb['reviews_per_month'].isna()]


# In[ ]:


airbnb['reviews_per_month'].fillna(0,inplace=True)


# In[ ]:


airbnb.drop('last_review',axis=1,inplace=True)


# In[ ]:


airbnb.tail()


# In[ ]:


sns.distplot(airbnb['price'])


# In[ ]:


sns.distplot(np.log1p(airbnb['price']))


# In[ ]:


airbnb['price'] = np.log1p(airbnb['price'])


# In[ ]:


sns.distplot(airbnb['minimum_nights'])


# In[ ]:


airbnb['minimum_nights'] = np.log1p(airbnb['minimum_nights'])


# In[ ]:


sns.distplot(airbnb['minimum_nights'])


# In[ ]:


plt.figure(figsize=(12,10))
sns.heatmap(airbnb.corr(),annot=True)


# In[ ]:


sns.countplot(airbnb['neighbourhood_group'])


# In[ ]:


sns.scatterplot(airbnb.latitude,airbnb.longitude)


# In[ ]:


sns.distplot(airbnb['number_of_reviews'])


# In[ ]:


sns.distplot(np.log1p(airbnb['number_of_reviews']))


# In[ ]:


sns.countplot(airbnb['room_type'])


# In[ ]:


airbnb['room_type'] = LabelEncoder().fit_transform(airbnb['room_type'])


# In[ ]:


airbnb['number_of_reviews'] = np.log1p(airbnb['number_of_reviews'])


# In[ ]:


airbnb.drop('calculated_host_listings_count',axis=1,inplace=True)


# In[ ]:


airbnb.drop(['id','host_id','reviews_per_month'],axis=1,inplace=True)


# In[ ]:


dummydata = pd.get_dummies(airbnb)


# In[ ]:


sc = StandardScaler()


# In[ ]:


scaledData = pd.DataFrame(sc.fit_transform(dummydata),columns=dummydata.columns)
scaledData.head()


# In[ ]:


x = scaledData.drop('price',axis=1)
y = scaledData['price']


# In[ ]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.30)


# In[ ]:


lr = LinearRegression()


# In[ ]:


lr.fit(xtrain,ytrain)
ypred = lr.predict(xtest)
ypred


# In[ ]:


r2_score(ytest,ypred)


# In[ ]:


np.sqrt(mean_squared_error(ytest,ypred))


# In[ ]:


r = Ridge()


# In[ ]:


r.fit(xtrain,ytrain)
ypredR = r.predict(xtest)


# In[ ]:


r2_score(ytest,ypredR)


# In[ ]:


np.sqrt(mean_squared_error(ytest,ypredR))


# In[ ]:


l = Lasso()


# In[ ]:


l.fit(xtrain,ytrain)


# In[ ]:


ypredLasso = l.predict(xtest)


# In[ ]:


r2_score(ytest,ypredLasso)


# In[ ]:


np.sqrt(mean_squared_error(ytest,ypredLasso))


# In[ ]:


e = ElasticNet()


# In[ ]:


e.fit(xtrain,ytrain)


# In[ ]:


ypredEnet = e.predict(xtest)


# In[ ]:


r2_score(ytest,ypredEnet)


# In[ ]:


np.sqrt(mean_squared_error(ytest,ypredEnet))


# In[ ]:


params = {
    'alpha' : [0.1,0.01,0.5,1,2,5,0.02,10,50],
    'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
    'random_state' : [0,1,123,5,10]
}
params_lasso = {
    'alpha' : [0.1,0.01,0.5,1,2,5,0.02,10,50,25,100],
    'random_state' : [0,1,123,5,10]
}
params_enet = {
    'alpha' : [0.1,0.01,0.5,1,2,5,0.02,10,50,25,100],
    'random_state' : [0,1,123,5,10]
}


# In[ ]:


gridR = GridSearchCV(estimator=r,param_grid=params,cv =3)


# In[ ]:


#gridR.fit(xtrain,ytrain)

