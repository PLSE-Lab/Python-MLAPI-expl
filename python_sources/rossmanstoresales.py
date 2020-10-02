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


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


store = pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')
data =  pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv')


# In[ ]:


print(data.shape,store.shape)


# In[ ]:


store.head()


# In[ ]:


data.head(20)


# In[ ]:


test = pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')


# In[ ]:


test.shape


# In[ ]:


test.head()


# In[ ]:


data.dtypes


# In[ ]:


data.describe(include='object')


# In[ ]:


data.describe()[['Sales','Customers']].loc['mean']


# In[ ]:


data.describe()[['Sales','Customers']].loc['max']


# In[ ]:


data.describe()[['Sales','Customers']].loc['min']


# In[ ]:


data.head()


# In[ ]:


data.Store.nunique()


# In[ ]:


data.Store.value_counts()


# In[ ]:


data.Store.value_counts().tail(50).plot.bar()


# In[ ]:


data.Store.value_counts().head(50).plot.bar()


# In[ ]:


data.DayOfWeek.value_counts()


# In[ ]:


data.Open.value_counts()


# In[ ]:


data[data['Customers']==data['Customers'].max()]


# In[ ]:


data[data['Sales']==data['Sales'].max()]


# In[ ]:


data.isnull().sum()


# In[ ]:


store.isnull().sum()


# In[ ]:


data['Date'] = pd.to_datetime(data['Date'],format = '%Y-%m-%d')

#for a single store

store_id = data.Store.unique()[0]
store_rows = data[data['Store']==store_id]
store_rows.resample('1d',on='Date')['Sales'].sum().plot.line(figsize=(10,8))


# In[ ]:


store_rows[store_rows.Sales==0]


# In[ ]:


test['Date'] = pd.to_datetime(test['Date'],format = '%Y-%m-%d')
store_test_rows = test[test['Store']==store_id]
store_test_rows['Date'].min(), store_test_rows['Date'].max()


# In[ ]:


store_rows['Sales'].plot.hist()


# In[ ]:


data['Sales'].plot.hist()


# In[ ]:


store[store['Store']==store_id].T


# In[ ]:


store[~store['Promo2SinceYear'].isna()].iloc[0]


# ## Missing value

# In[ ]:


store.isna().sum()


# In[ ]:


#Method1

store['Promo2SinceWeek'] = store['Promo2SinceWeek'].fillna(0)
store['Promo2SinceYear'] = store['Promo2SinceYear'].fillna(store['Promo2SinceYear'].mode().iloc[0])
store['PromoInterval'] = store['PromoInterval'].fillna(store['PromoInterval'].mode().iloc[0])

store['CompetitionDistance'] = store['CompetitionDistance'].fillna(store['CompetitionDistance'].max())
store['CompetitionOpenSinceMonth'] = store['CompetitionOpenSinceMonth'].fillna(store['CompetitionOpenSinceMonth'].mode().iloc[0])
store['CompetitionOpenSinceYear'] = store['CompetitionOpenSinceYear'].fillna(store['CompetitionOpenSinceYear'].mode().iloc[0])


# In[ ]:


store.isna().sum()


# In[ ]:


data_merged = data.merge(store, on ='Store', how='left')


# In[ ]:


data_merged


# In[ ]:


data.shape, data_merged.shape


# In[ ]:


data_merged.isna().sum()


# In[ ]:


data_merged.dtypes


# ## Encoding
# 

# In[ ]:


data_merged['day'] = data_merged['Date'].dt.day
data_merged['year'] = data_merged['Date'].dt.year
data_merged['month'] = data_merged['Date'].dt.month
#data_merged['Date'].dt.strftime('%a')


# In[ ]:


data_merged['StateHoliday'].unique()
data_merged['StateHoliday'] = data_merged['StateHoliday'].map({'0':0, 0:0,'a':1,'b':2,'c':3}).astype(int)
data_merged['StoreType'] = data_merged['StoreType'].map({'c':0, 'a':1,'d':2,'b':3}).astype(int)
data_merged['Assortment'] = data_merged['Assortment'].map({'a':0, 'c':1,'b':2}).astype(int)
data_merged['PromoInterval'] = data_merged['PromoInterval'].map({'Jan,Apr,Jul,Oct':0, 'Feb,May,Aug,Nov':1,'Mar,Jun,Sept,Dec':2}).astype(int)
data_merged['StateHoliday']


# In[ ]:


data_merged.dtypes


# In[ ]:


data_merged.shape


# ## Train & Validate Split

# In[ ]:


from sklearn.model_selection import train_test_split
import numpy as np
X = data_merged.drop(['Sales','Date','Customers'],axis=1)
y = data_merged['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, np.log(y+1), test_size=0.3, random_state=1)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
model = DecisionTreeRegressor(max_depth=11)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
r2_score(y_pred,y_test)

# def ToWeight(y):
#     w = np.zeros(y.shape, dtype=float)
#     ind = y != 0
#     w[ind] = 1./(y[ind]**2)
#     return w

# def rmspe(y, yhat):
#     w = ToWeight(y)
#     rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
#     return rmspe

# rmse_val = np.sqrt(mean_squared_error(y_test_inv,y_pred_inv))
# rmspe_val = rmspe(y_test_inv,y_pred_inv)
# print(rmse_val,rmspe_val)


# In[ ]:



# def draw_tree(model, columns):
#     import pydotplus
#     from sklearn.externals.six import StringIO
#     from IPython.display import Image
#     import os
#     from sklearn import tree
    
#     graphviz_path = 'C:\Program Files (x86)\Graphviz2.38/bin/'
#     os.environ["PATH"] += os.pathsep + graphviz_path

#     dot_data = StringIO()
#     tree.export_graphviz(model,
#                          out_file=dot_data,
#                          feature_names=columns)
#     graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#     return Image(graph.create_png())


# In[ ]:


#draw_tree(model,X)


# In[ ]:


y_test_inv = np.exp(y_test) - 1
y_pred_inv = np.exp(y_pred) - 1
np.sqrt(mean_squared_error(y_test_inv,y_pred_inv))


# In[ ]:


r2_score(y_test_inv,y_pred_inv)


# In[ ]:


test.head()


# In[ ]:


pd.Series(model.feature_importances_,index=X.columns)


# In[ ]:


data_merged.corr()['Sales'].sort_values(ascending=False)


# ### customers is an important column so we cant drop it as the column in not present in test csv[](http://)

# In[ ]:


stores_avg_cuts = data.groupby(['Store'])[['Customers']].mean().reset_index().astype(int)
test1 = test.merge(stores_avg_cuts,
                  on = 'Store',
                  how = 'left')

test_merged = test1.merge(store,
                          on='Store',
                          how='inner')
test_merged['Open'] = test_merged['Open'].fillna(1) 
test_merged['Date'] = pd.to_datetime(test_merged['Date'], format = '%Y-%m-%d')
test_merged['day'] = test_merged['Date'].dt.day
test_merged['month'] = test_merged['Date'].dt.month
test_merged['year'] = test_merged['Date'].dt.year


# In[ ]:


test_merged.dtypes


# In[ ]:



test_merged['StateHoliday'] = test_merged['StateHoliday'].map({'0':0,'a':1}).astype(int)
test_merged['StoreType'] = test_merged['StoreType'].map({'c':0, 'a':1,'d':2,'b':3}).astype(int)
test_merged['Assortment'] = test_merged['Assortment'].map({'a':0, 'c':1,'b':2}).astype(int)
test_merged['PromoInterval'] = test_merged['PromoInterval'].map({'Jan,Apr,Jul,Oct':0, 'Feb,May,Aug,Nov':1,'Mar,Jun,Sept,Dec':2}).astype(int)


# In[ ]:


test_merged.shape


# In[ ]:


test_pred = model.predict(test_merged[X.columns])
test_pred_inv = np.exp(test_pred) -1
submission_predicted = pd.DataFrame({'Id': test['Id'], 'Sales': test_pred_inv })
submission_predicted.to_csv('submission.csv',index=False)


# # Optimisation-1
# 
# ## Hyperparameter Tuning

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


# parameters = {'max_depth' : list(range(5,20))}
# base_model = DecisionTreeRegressor()
# cv_model = GridSearchCV(base_model, param_grid=parameters, cv=5, return_train_score=True).fit(X_train,y_train)


# In[ ]:


# cv_model.best_params_


# In[ ]:


# dv_cv_results = pd.DataFrame(cv_model.cv_results_).sort_values(by='mean_test_score', ascending=False)
# dv_cv_results.set_index('param_max_depth')['mean_test_score'].plot.line()
# dv_cv_results.set_index('param_max_depth')['mean_train_score'].plot.line()
# plt.legend(['Test Scores', 'Train Scores'])


# In[ ]:


# dv_cv_results


# In[ ]:


# X_1 = X.drop('Customers',axis=1)


# In[ ]:


# test_pred = model.predict(test_merged[X_1.columns])
# test_pred_inv = np.exp(test_pred) -1
# submission_predicted = pd.DataFrame({'Id': test['Id'], 'Sales': test_pred_inv })
# submission_predicted.to_csv('submission.csv',index=False)


# In[ ]:




