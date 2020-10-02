#!/usr/bin/env python
# coding: utf-8

# To be able to use our XGBRegressor regression, we need to load it

# In[ ]:


get_ipython().system('pip install xgboost')


# We add our required libraries

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


# We do this to avoid unnecessary errors

# In[ ]:


from warnings import filterwarnings
filterwarnings("ignore")


# We take our data set

# In[ ]:


df = pd.read_csv("../input/kc-housesales-data/kc_house_data.csv")


# Let's look at the first 5 lines of our data set

# In[ ]:


df.head()


# Let's get more detailed information about our data set

# In[ ]:


df.info()


# Let's take blank, unnecessary data in the dataset

# In[ ]:


df = df.dropna()


# Let's throw the pillars we don't need

# In[ ]:


df.drop(["id","date"], axis = 1, inplace = True)


# Let's write our function that will try each of our regression

# In[ ]:


def compML(df, y, alg):
    
    y = df.price.values.reshape(-1,1)
    x = df.drop(["price"], axis = 1)
    
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    y = scaler.fit_transform(y)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y , test_size = 0.2, random_state = 0, shuffle=True)
    
    model = alg().fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    r2 = r2_score (y_test, y_pred)
    
    model_ismi = alg.__name__
    
    print(model_ismi, "R2_Score ---> ", r2)


# Let's write the regression names we will give below the loop

# In[ ]:


models = [LinearRegression,
          DecisionTreeRegressor, 
          KNeighborsRegressor, 
          MLPRegressor, 
          RandomForestRegressor, 
          GradientBoostingRegressor,
          SVR,
          XGBRegressor]


# wrote the results of our regressions

# In[ ]:


for i in models:
    compML(df, "price", i)


# our best regressions ---> XGBRegressor, RandomForestRegressor  
