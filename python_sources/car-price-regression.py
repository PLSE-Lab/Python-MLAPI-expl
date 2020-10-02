#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn import model_selection 


# In[ ]:


df = pd.read_csv('../input/vehicle-dataset-from-cardekho/car data.csv')


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.Seller_Type.unique()


# In[ ]:


df.Fuel_Type.unique()


# In[ ]:


df.Transmission.unique()


# In[ ]:


df.Owner.unique()


# In[ ]:


df.Year.unique()


# In[ ]:


df.drop(['Car_Name'],axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df['current year']=2020


# In[ ]:


df.head()


# In[ ]:


df['total_year'] = df['current year'] - df['Year']


# In[ ]:


df.drop(['Year','current year'],axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df = df.sample(frac=1).reset_index(drop=True) 


# In[ ]:


df.head()


# In[ ]:


df = pd.get_dummies(df,drop_first=True)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[ ]:


X=df.drop(['Selling_Price'],axis=1)
y=df['Selling_Price']


# In[ ]:


y.shape


# In[ ]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


# we are tuning three hyperparameters right now, we are passing the different values for both parameters
grid_param = {
    "n_estimators" : [120,300,500,800,1200],
    'max_depth' : [5, 8, 15, 25, 30],
    'max_features' : ['auto','log2', 'sqrt'],
    'min_samples_split' : [1, 2, 5, 10, 15, 100],
    'min_samples_leaf' : [1, 2, 5, 10]
}
rand_reg = RandomForestRegressor()


# In[ ]:


grid_search = GridSearchCV(estimator=rand_reg, param_grid=grid_param, cv=5, n_jobs = -1, verbose = 3)


# In[ ]:


grid_search.fit(X_train,y_train)


# In[ ]:


grid_search.best_params_


# In[ ]:


rand_reg_new = RandomForestRegressor(
 max_depth= 8,
 max_features = 'auto',
 min_samples_leaf = 1,
 min_samples_split = 2,
 n_estimators = 120,
 random_state = 42)


# In[ ]:


rand_reg_new.fit(X_train,y_train)


# In[ ]:


predictions = rand_reg_new.predict(X_test)


# In[ ]:


from sklearn import metrics
import numpy as np

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:


import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rand_reg_new, file)

