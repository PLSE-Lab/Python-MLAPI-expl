#!/usr/bin/env python
# coding: utf-8

# Sales forecasting has always been one of the most predominant applications of machine learning. Big companies like Walmart have been employing this technique to achieve steady and enormous growth over decades now. In this challenge, you as a data scientist must use machine learning to help a small grocery store in predicting its future sales and making better business decisions.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Dataset

# In[ ]:


train = pd.read_csv("/kaggle/input/grocery-sales-forecast-weekend-hackathon/Grocery_Sales_ParticipantsData/Train.csv")
test = pd.read_csv("/kaggle/input/grocery-sales-forecast-weekend-hackathon/Grocery_Sales_ParticipantsData/Test.csv")
sample = pd.read_excel("/kaggle/input/grocery-sales-forecast-weekend-hackathon/Grocery_Sales_ParticipantsData/Sample_Submission.xlsx")


# In[ ]:


print(train.head())
train.shape


# In[ ]:


print(test.head())
test.shape


# In[ ]:


print(sample.head())
sample.shape


# In[ ]:


train.isnull().sum()


# # Univirate Analysis

# In[ ]:


import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
plt.boxplot(train['GrocerySales'])


# # Deleting outliers

# In[ ]:


# We are deleting the outliers from quartile 27th percentile and 73th percentile as appropriate for our model.
Q1 = train.quantile(0.27)
Q3 = train.quantile(0.73)
IQR = Q3 - Q1
print(IQR)


# In[ ]:


train.shape


# In[ ]:


train1 = train[~((train < (Q1 - 1.5 * IQR)) |(train > (Q3 + 1.5 * IQR))).any(axis=1)]
train1.shape


# In[ ]:


plt.boxplot(train1['GrocerySales'])


# In[ ]:


plt.scatter(train['Day'], train['GrocerySales'],  color='black')


# In[ ]:


fig= plt.figure(figsize=(15,5))
plt.plot(train1['Day'], train1['GrocerySales'], color='green',markersize=1)


# # Select features and labels

# In[ ]:


#Fetaures 
X = train1['Day'].copy()
X.head()


# In[ ]:


y = train1['GrocerySales'].copy()
y.head()


# In[ ]:


X = np.array(train1['Day'])
X=X.reshape(-1, 1)
X


# # Split the data

# In[ ]:


from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,train_size=0.8, test_size=0.2,random_state = 0)


# # Let's try different algorithms and check the mean square error

# # LinearRegression

# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(train_X,train_y)


# In[ ]:


predict = regr.predict(val_X)
print('Mean squared error: %.2f'
      % mean_squared_error(val_y,predict))


# In[ ]:


plt.scatter(train['Day'], train['GrocerySales'],  color='black')
plt.plot(val_X, predict, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


# # KNeighborsRegressor

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor

neigh = KNeighborsRegressor()
neigh.fit(train_X,train_y)
predict_n = neigh.predict(val_X)
print('Mean squared error: %.2f'
     % mean_squared_error(val_y,predict_n))


# ## Tuning of KNeighborsRegressor

# In[ ]:


parameters_for_testing = {
   'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
   'n_neighbors':[1,4,2],
   'weights':['uniform','distance'],
    
}


# In[ ]:


from sklearn.model_selection import GridSearchCV
gsearch1 = GridSearchCV(cv=8,estimator = neigh, param_grid = parameters_for_testing, n_jobs=-1,verbose=0,scoring='neg_mean_squared_error')
gsearch1.fit(train_X,train_y)


# In[ ]:


print (gsearch1.best_params_)


# In[ ]:


print (gsearch1.best_score_)


# In[ ]:


predict_grid = gsearch1.predict(val_X)
print('Mean squared error: %.2f'
     % np.sqrt(mean_squared_error(val_y,predict_grid)))


# # DecisionTreeRegressor

# In[ ]:


from sklearn import tree
clf = tree.DecisionTreeRegressor()
clf.fit(train_X,train_y)
predict_clf = clf.predict(val_X)
print('Mean squared error: %.2f'
     % mean_squared_error(val_y,predict_clf))


# ## Tuning of DecisionTreeRegressor

# In[ ]:


parameters = {
   'criterion':['mse', 'friedman_mse', 'mae'],
   'splitter':['best', 'random'],
   'min_samples_split':[2,3,4,5]
}


# In[ ]:


gsearch2 = GridSearchCV(estimator = clf, param_grid = parameters, n_jobs=-1,verbose=0,scoring='neg_mean_squared_error')
gsearch2.fit(train_X,train_y)


# In[ ]:


print('best params')
print (gsearch2.best_params_)
print('best score')
print (gsearch2.best_score_)


# In[ ]:


predict_grid2 = gsearch2.predict(val_X)
print('Mean squared error: %.2f'
     % (mean_squared_error(val_y,predict_grid2)))


# # XGBOOST

# In[ ]:


import xgboost as xgb
xg = xgb.XGBRegressor(eta=0.6599999999999999999999999999)
xg = xg.fit(train_X,train_y)
predict_x = xg.predict(val_X)
print('Mean squared error is: {}'.format((mean_squared_error(val_y,predict_x))))


# # Ensembling Methods

# # 1. Voting

# In[ ]:


from sklearn.ensemble import VotingRegressor
ereg = VotingRegressor(estimators=[('nh', neigh), ('tree', gsearch1)])
ereg = ereg.fit(train_X,train_y)
predict_clf = ereg.predict(val_X)
print('Mean squared error: %.2f'
     % mean_squared_error(val_y,predict_clf))


# # 2. Bagging

# In[ ]:


from sklearn.ensemble import BaggingRegressor
bag = BaggingRegressor(base_estimator=xg, n_estimators=14, random_state=0).fit(train_X,train_y)
predict_bag = bag.predict(val_X)
print('Mean squared error is {}'.format((mean_squared_error(val_y,predict_bag))))


# In[ ]:


plt.scatter(train_X, train_y, color = 'blue') 
  
plt.plot(train_X, bag.predict(train_X), color = 'red') 
plt.title('Bagging meta-estimator') 
plt.xlabel('Days') 
plt.ylabel('Sale') 
  
plt.show() 


# # CatBoostRegressor

# In[ ]:


from catboost import CatBoostRegressor

model = CatBoostRegressor()
#train the model
model.fit(train_X,train_y)
# make the prediction using the resulting model
preds = model.predict(val_X)
print('Mean squared error: %.2f'
     % (mean_squared_error(val_y,preds)))


# # lightgbm

# In[ ]:


import lightgbm as lgb 
lg = lgb.LGBMRegressor()
lg = lg.fit(train_X,train_y)
predict_l = lg.predict(val_X)
print('Mean squared error: %.2f'
     % (mean_squared_error(val_y,predict_l)))


# In[ ]:


test = np.array(test)
test=test.reshape(-1, 1)
test


# In[ ]:


bag = BaggingRegressor(base_estimator=xg, n_estimators=108, random_state=0).fit(X,y)


# In[ ]:


predict = bag.predict(test)


# In[ ]:


df = pd.DataFrame(predict,columns=['GrocerySales'])
df.head()


# In[ ]:


df.to_excel("submission2.xlsx")


# In[ ]:




