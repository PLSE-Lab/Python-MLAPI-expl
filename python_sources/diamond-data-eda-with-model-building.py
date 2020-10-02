#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df=pd.read_csv('/kaggle/input/diamonds/diamonds.csv')


# In[ ]:


df.head()


# In[ ]:


df=df.drop(['Unnamed: 0'],axis=1)


# In[ ]:


df.isnull().sum()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


print('X-Values with zero value ',df['x'].value_counts()[0])
print('Y-Values with zero value ',df['y'].value_counts()[0])
print('Z-Values with zero value ',df['z'].value_counts()[0])


# In[ ]:


df.loc[df['x']==0]


# In[ ]:


df[['x','y','z']]=df[['x','y','z']].replace(0,np.NaN)


# In[ ]:


df.isnull().sum()


# In[ ]:


df['x'].describe()


# In[ ]:


df.plot(kind='box',layout=(3,3),subplots=True,figsize=(15,10),title=['Carat','Depth','Table','Price','X','Y','Z'],)


# In[ ]:


cat=list(df.select_dtypes(include='object').columns)
numeric=list(df.select_dtypes(exclude='object').columns)


# In[ ]:


def iqr(p):
    a=[]
    q1=df[p].quantile(0.25)
    q3=df[p].quantile(0.75)
    iqr=q3-q1
    ulim=q3+(1.5*iqr)
    llim=q1-(1.5*iqr)   
    for i in df[p]:
        if (i>ulim) or (i<llim):
            i=np.NaN
        else:
            i=i
        a.append(i)
    return (a)


# In[ ]:


for w in numeric:
    df[w]=iqr(w)   


# In[ ]:


df.isna().sum()


# In[ ]:


df.plot(kind='box',layout=(3,3),subplots=True,figsize=(15,10),title=['Carat','Depth','Table','Price','X','Y','Z'],)


# In[ ]:


df.describe()


# In[ ]:


cat=df.select_dtypes(include='object')
numeric=df.select_dtypes(exclude='object')


# In[ ]:


for i in numeric:
    df[i]=df[i].fillna(df[i].mean())


# In[ ]:


df.isna().sum()


# In[ ]:


sns.set()
f,ax=plt.subplots(1,3,figsize=(15,5))
sns.countplot(df['color'],ax=ax[0])
sns.countplot(df['clarity'],ax=ax[1])
sns.countplot(df['cut'],ax=ax[2])


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


le=LabelEncoder()


# In[ ]:


cat_01=cat.apply(le.fit_transform)


# In[ ]:


df=df.drop(cat,axis=1)


# In[ ]:


df=pd.concat([df,cat_01],axis=1)


# In[ ]:


plt.figure(figsize=(10,7))
sns.heatmap(df.corr(),annot=True,cmap='YlGnBu')


# # Statistical Method

# In[ ]:


import statsmodels.api as sm


# In[ ]:


X=df.drop('price',axis=1)
y=df['price']


# In[ ]:


y.shape


# In[ ]:


xc=sm.add_constant(X)


# In[ ]:


lin_reg=sm.OLS(y,xc).fit()


# In[ ]:


lin_reg.summary()


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


vif=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
pd.DataFrame({'vif':vif},index=X.columns)


# In[ ]:


X=df.drop(['price','depth'],axis=1)
y=df['price']
xc=sm.add_constant(X)
lin_reg=sm.OLS(y,xc).fit()
lin_reg.summary()


# In[ ]:


vif=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
pd.DataFrame({'vif':vif},index=X.columns)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X=df.drop(['price'],axis=1)
y=df['price']


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[ ]:


lr=LinearRegression()


# In[ ]:


lr.fit(X_train,y_train)


# In[ ]:


y_pred=lr.predict(X_test)


# In[ ]:


r2_score(y_test,y_pred)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor()


# In[ ]:


rfr.fit(X_train,y_train)


# In[ ]:


y_pred=rfr.predict(X_test)


# In[ ]:


r2_score(y_test,y_pred)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


n_estimators=np.arange(1,200,2)
max_depth=np.arange(10,100,10)
min_samples_split=[2,3,4,5,6,7,8,9,10]
min_samples_leaf=[2,3,4,5,6,7,8,9,10]
random_state=np.arange(1,100,1)
param_grid={'n_estimators':n_estimators,'max_depth':max_depth,'min_samples_leaf':min_samples_leaf,'min_samples_split':min_samples_split,
            'random_state':random_state}


# In[ ]:


rfr_gs=RandomizedSearchCV(estimator=rfr,param_distributions=param_grid,scoring='r2').fit(X_train,y_train)


# In[ ]:


rfr_gs.best_estimator_


# In[ ]:


rfr_gs.best_params_


# In[ ]:


y_pred=rfr_gs.predict(X_test)


# In[ ]:


r2_score(y_test,y_pred)


# In[ ]:


from sklearn.feature_selection import RFE


# In[ ]:


rfe=RFE(lr,6)


# In[ ]:


rfe.fit(X_train,y_train)


# In[ ]:


rfe.ranking_


# In[ ]:


pd.DataFrame(list(zip(X.columns,rfe.ranking_)),columns=['X','Ranking'])


# In[ ]:


#no of features
nof_list=np.arange(1,10)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))

