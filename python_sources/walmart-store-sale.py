#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


walmart_features = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv')
walmart_features.head()
walmart_features.shape


# In[ ]:


walmart_sales = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv')
walmart_sales.head()
walmart_sales.shape


# In[ ]:


store_sales = walmart_sales.groupby('Store').mean()
#store_sales.set_index('Dept', inplace =True)
#store_sales['Weekly_Sales'].plot(kind = 'bar',figsize = (17,10))
plt.figure(figsize = (15,10))
plt.title('Store Wise Weekly Sales', fontsize = 15)
barplot = sns.barplot(x = store_sales.index, y = 'Weekly_Sales', data = store_sales)
for x in barplot.patches:
    barplot.annotate(format(x.get_height(), '.2f'), (x.get_x() + x.get_width() / 2., x.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# In[ ]:


saleon_holiday =walmart_sales[['Weekly_Sales', 'Date','Store']][walmart_sales['IsHoliday']==1]
saleon_holiday['Date'] = pd.to_datetime(saleon_holiday['Date'])
saleon_holiday.set_index('Date', inplace = True)
saleon_holiday['Week'] = saleon_holiday.index.week
saleon_holiday['Month'] = saleon_holiday.index.month


# In[ ]:


month_wise = saleon_holiday.groupby('Month').mean()
month_wise['Sales'] = month_wise['Weekly_Sales'].copy()
month_wise.drop('Weekly_Sales',axis = 1, inplace = True)
plt.title("Monthly Sales During Holiday's")
sns.barplot(x = month_wise.index, y = 'Sales', data = month_wise)


# In[ ]:


wal = walmart_features[['Date','MarkDown1', 'MarkDown2','MarkDown3', 'MarkDown4', 'MarkDown5']]
wal.head()


# In[ ]:


new_data = pd.merge(walmart_sales, wal, on = 'Date')
new_data.tail(100)


# In[ ]:


heat = new_data[new_data['IsHoliday']==True].corr()
sns.heatmap(heat, annot = True, fmt = '.2f',cmap = 'coolwarm')


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import r2_score
from math import sqrt


# In[ ]:


test_set = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv')
test_set.head()


# In[ ]:


trained = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv')
W_Sales= trained['Weekly_Sales']


# In[ ]:


trainer = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv')
trainer['Date'] = pd.to_datetime(trainer['Date'])
trainer['Day'] = trainer['Date'].dt.day
trainer['Month'] = trainer['Date'].dt.month
trainer['Year'] = trainer['Date'].dt.year
trainer.head()


# In[ ]:


trainer.drop('Date', axis = 1, inplace = True)


# In[ ]:


trainer.drop('Weekly_Sales',axis = 1, inplace = True)


# In[ ]:


test_set['Date'] = pd.to_datetime(test_set['Date'])
test_set['Day'] = test_set['Date'].dt.day
test_set['Month'] = test_set['Date'].dt.month
test_set['Year'] = test_set['Date'].dt.year


# In[ ]:


Date = test_set['Date']


# In[ ]:


test_set.drop('Date', axis =1, inplace = True)


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(trainer, W_Sales, test_size = 0.3, random_state = 42)


# In[ ]:


import catboost as cd
from sklearn.model_selection import RandomizedSearchCV
cdr = cd.CatBoostRegressor()
para = {'iterations' : [100,200,300,500],
       'depth': [2,4,5,6,8,9],
       'learning_rate': [0.03,0.1,0.15,0.2,0.3,0.01]}
finclf = RandomizedSearchCV(estimator = cdr, param_distributions = para, cv = 5, n_jobs = -1)
finclf.fit(X_train, Y_train)


# In[ ]:


finclf.best_params_


# In[ ]:


ca_bo_re = cd.CatBoostRegressor(iterations = 200, depth = 8, learning_rate = 0.3)
ca_bo_re.fit(X_train,Y_train)
predic = ca_bo_re.predict(X_test)


# In[ ]:


print(sqrt(mean_squared_error(Y_test, predic)))
print(r2_score(Y_test,predic))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
paran = {'n_estimators': [100,150,200,300],
        'max_depth': [2,3,5,6,7,8,9, None]}
clfer = RandomizedSearchCV(estimator = rfr,
                          param_distributions = paran,
                          cv = 5, n_jobs = -1)
clfer.fit(X_train,Y_train)


# In[ ]:


clfer.best_params_


# In[ ]:


ranfo = RandomForestRegressor(n_estimators = 200, max_depth = 9)
ranfo.fit(X_train,Y_train)
pres = ranfo.predict(X_test)
print(sqrt(mean_squared_error(Y_test, pres)))
print(r2_score(Y_test,pres))


# In[ ]:


rafor = RandomForestRegressor(n_estimators = 100)
rafor.fit(X_train,Y_train)
pr = rafor.predict(X_test)
print(sqrt(mean_squared_error(Y_test,pr)))
print(r2_score(Y_test,pr))


# In[ ]:


from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train,Y_train)
predu = xgb.predict(X_test)
print(sqrt(mean_squared_error(Y_test,predu)))


# In[ ]:


r2_score(Y_test,predu)


# In[ ]:


final2 = rafor.predict(test_set)


# In[ ]:


Store = test_set['Store']
Dept = test_set['Dept']


# In[ ]:


solution = pd.DataFrame(Dept)
solution['Weekly_Sales'] = final2
solution['ID'] = test_set['Store'].astype(str)+'_'+test_set['Dept'].astype(str)+'_'+Date.astype(str)
solutions = solution[['ID','Weekly_Sales']]


# In[ ]:


solutions.to_csv('submission.csv', index = False)

