#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv('../input/Train.csv')


# In[ ]:


data.isnull().sum()


# In[ ]:


data.head()


# In[ ]:


data['Item_Fat_Content'].value_counts()


# In[ ]:


data.loc[1,'Item_Fat_Content'],range(data.shape[0])


# In[ ]:



for i in range(data.shape[0]):
    if(data.loc[i,'Item_Fat_Content']  in ('LF','low fat','Low Fat')):
        data.loc[i,'Item_Fat_Content']='Low Fat'
        
    else:
        data.loc[i,'Item_Fat_Content']='Regular'
        
            


# In[ ]:


data['Item_Fat_Content'].value_counts()


# In[ ]:


data['Item_Weight'].fillna((data['Item_Weight'].mean()),inplace=True)


# In[ ]:


data['Outlet_Size'].fillna(data['Outlet_Size'].mode().iloc[0],inplace=True)


# In[ ]:


data['Outlet_Size'].value_counts().plot(kind='bar')


# In[ ]:


corr=data.apply(lambda x:pd.factorize(x)[0]).corr(method='pearson')


# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(corr,annot=True)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[ ]:


data['Outlet_Location_Type']=le.fit_transform(data['Outlet_Location_Type'])
data['Item_Identifier']=le.fit_transform(data['Item_Identifier'])
data['Item_Fat_Content']=le.fit_transform(data['Item_Fat_Content'])
data['Item_Type']=le.fit_transform(data['Item_Type'])
data['Outlet_Type']=le.fit_transform(data['Outlet_Type'])
data['Outlet_Establishment_Year'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Size']=le.fit_transform(data['Outlet_Size'])


# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(data.corr(),annot=True)


# In[ ]:


x=data.iloc[:,:-1]
y=data.loc[:,'Item_Outlet_Sales']


# In[ ]:


x.head()


# In[ ]:


y.plot(kind='box')


# In[ ]:


np.log(y).plot(kind='box')


# In[ ]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[ ]:


dummy_x=pd.get_dummies(x)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,RobustScaler


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(dummy_x,np.log(y))


# In[ ]:


scaler=RobustScaler()


# In[ ]:


X_train_Scaled = scaler.fit(x_train).transform(x_train)
X_test_Scaled = scaler.fit(x_test).transform(x_test)


# In[ ]:


#lr.fit(x_train,y_train)
lr.fit(X_train_Scaled,y_train)


# In[ ]:


#lr.score(x_train,y_train),lr.score(x_test,y_test)
lr.score(X_train_Scaled,y_train),lr.score(X_test_Scaled,y_test)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()


# In[ ]:


rf.fit(x_train,y_train)


# In[ ]:


rf.score(x_train,y_train),rf.score(x_test,y_test)


# In[ ]:


param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}


# In[ ]:


from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator =rf , param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)


# In[ ]:


#grid_search.fit(x_train,y_train)


# In[ ]:


#grid_search.best_params_


# In[ ]:


rf1=RandomForestRegressor(bootstrap= True,
 max_depth= 100,
 max_features= 3,
 min_samples_leaf=3,
 min_samples_split=8,
 n_estimators=300)


# In[ ]:


rf1.fit(x_train,y_train)


# In[ ]:


rf1.score(x_train,y_train),rf1.score(x_test,y_test)


# In[ ]:


testdata=pd.read_csv('../input/Test.csv')


# In[ ]:


testdata['Item_Fat_Content'].value_counts()


# In[ ]:


for i in range(testdata.shape[0]):
    if(testdata.loc[i,'Item_Fat_Content']  in ('LF','low fat','Low Fat')):
        testdata.loc[i,'Item_Fat_Content']='Low Fat'
        
    else:
        testdata.loc[i,'Item_Fat_Content']='Regular'
        


# In[ ]:


testdata.isnull().sum()


# In[ ]:


testdata['Item_Weight'].fillna((testdata['Item_Weight'].mean()),inplace=True)
testdata['Outlet_Size'].fillna(testdata['Outlet_Size'].mode().iloc[0],inplace=True)


# In[ ]:


testingItemIdentifier =  testdata['Item_Identifier'].values
testingOutletIdentifier =  testdata['Outlet_Identifier'].values


# In[ ]:


testdata['Outlet_Location_Type']=le.fit_transform(testdata['Outlet_Location_Type'])
testdata['Item_Identifier']=le.fit_transform(testdata['Item_Identifier'])
testdata['Item_Fat_Content']=le.fit_transform(testdata['Item_Fat_Content'])
testdata['Item_Type']=le.fit_transform(testdata['Item_Type'])
testdata['Outlet_Type']=le.fit_transform(testdata['Outlet_Type'])
testdata['Outlet_Establishment_Year'] = 2013 - testdata['Outlet_Establishment_Year']
testdata['Outlet_Size']=le.fit_transform(testdata['Outlet_Size'])


# In[ ]:


testdata.head()


# In[ ]:


testdata=pd.get_dummies(testdata)
testScaled = scaler.fit(testdata).transform(testdata)


# In[ ]:


#predictions=rf.predict(testdata)
predictions2=lr.predict(testScaled)


# In[ ]:


#predictions_org=np.exp(predictions)
predictions2_org=np.exp(predictions2)


# In[ ]:





# In[ ]:


sub = pd.DataFrame({'Item_Identifier' : testingItemIdentifier, 'Outlet_Identifier' : testingOutletIdentifier,
                    'Item_Outlet_Sales' : predictions_org})
sub.to_csv('submission4.csv', index=False)


# In[ ]:


sub = pd.DataFrame({'Item_Identifier' : testingItemIdentifier, 'Outlet_Identifier' : testingOutletIdentifier,
                    'Item_Outlet_Sales' : predictions2_org})
sub.to_csv('submission5.csv', index=False)


# In[ ]:


from xgboost import XGBRegressor


# In[ ]:


xgmodel=XGBRegressor()


# In[ ]:


xgmodel.fit(x_train,y_train)


# In[ ]:


xgmodel.score(x_train,y_train),xgmodel.score(x_test,y_test)


# In[ ]:


xgb=XGBRegressor(n_estimators=500, learning_rate=0.05)


# In[ ]:


xgb.fit(x_train,y_train)


# In[ ]:


xgb.score(x_train,y_train),xgb.score(x_test,y_test)


# In[ ]:


import xgboost as xgb
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)


# In[ ]:


params = {
    # Parameters that we are going to tune.
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    'objective':'reg:linear',
}


# In[ ]:


params['eval_metric'] = "mae"


# In[ ]:


num_boost_round = 999


# In[ ]:


model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10
)


# In[ ]:


cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=42,
    nfold=5,
    metrics={'mae'},
    early_stopping_rounds=10
)


# In[ ]:


cv_results


# In[ ]:


model.best_iteration


# In[ ]:


dtest1=xgb.DMatrix(testdata)


# In[ ]:


xgprediction=model.predict(dtest1)
testdata.head()


# In[ ]:


xgprediction1=np.exp(xgprediction)


# In[ ]:


sub = pd.DataFrame({'Item_Identifier' : testingItemIdentifier, 'Outlet_Identifier' : testingOutletIdentifier,
                    'Item_Outlet_Sales' : xgprediction1})
sub.to_csv('submission6.csv', index=False)


# In[ ]:




