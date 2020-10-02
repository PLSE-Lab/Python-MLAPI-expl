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



# Modules for importing data 
import pandas as pd
import numpy as np

# Modules for data visualizaion 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Modules for regression 
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score


# In[ ]:





# In[ ]:


import pandas as pd
border_cross = pd.read_csv("../input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv")


# In[ ]:


# basic checks and data cleaning
for i  in border_cross.head():
    print(i)


# In[ ]:


border_cross.describe()


# In[ ]:


border_cross.isnull().sum()


# In[ ]:


border_cross[["point", "lat long"]] = border_cross["Location"].str.split("(", n = 3, expand = True)
border_cross[["lati", "long"]] = border_cross["lat long"].str.split(")", n = 2, expand = True )
border_cross[["lat", "long"]] = border_cross["lati"].str.split(" ", n = 2, expand = True)
border_cross[["date", "time", "session"]] = border_cross["Date"].str.split(" ", n = 3, expand = True)
border_cross[["date", "month", "year"]] = border_cross["date"].str.split("/", n =3, expand = True)
b_cross = border_cross.drop(columns = ["Date", "session", "month", "time", "lati", "lat long", "point", "date", "Location"])


# In[ ]:


for i  in b_cross.head():
    print(i)


# In[ ]:


# Data analysis
# Border 
border = b_cross.groupby("Border")["Value"].sum().plot(kind = "bar", alpha = 0.5, color = ["r", "g"], edgecolor = "b")


# In[ ]:


# Top States
state = b_cross.groupby("State")["Value"].sum().sort_values(ascending = False).head(5).plot(kind = "bar", alpha = 0.5, color = ["r","g", "b", "y", "m"], edgecolor = "b")


# In[ ]:


# Top mode of transportation 
transport = b_cross.groupby("Measure")["Value"].sum().sort_values(ascending = False).head(5).plot(kind = "bar", alpha = 0.5, color = ["r","g", "b", "y", "m" ], edgecolor = "b")


# In[ ]:


# Top year
year = b_cross.groupby("year")["Value"].sum().sort_values(ascending = False).head(5).plot(kind = "bar", alpha = 0.5, color = ["r","g", "b", "y", "m"], edgecolor = "b")


# In[ ]:


# Creating train and test data set for regression  

b_cross1 = pd.get_dummies(b_cross[["year", "Port Name", "Border", "State", "Measure", "Value"]])


# In[ ]:


# Encoding categorical values
def Encode(b_cross):
    for column in b_cross.columns[b_cross.columns.isin(["Border", "State", "Measure"])]:
        b_cross[column] = b_cross[column].factorize()[0]
    return b_cross
b_cross_en = Encode(b_cross.copy())


# In[ ]:


ind_var = b_cross_en.drop(columns = ["Port Name", "Value", "Port Code", "long", "lat"])
tar_var = b_cross1["Value"]


# In[ ]:


for i in ind_var.head(5):
    print(i)


# In[ ]:


x = np.array(ind_var)
y = np.array(tar_var)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 52)


# In[ ]:



# Linear regression 
mod = LinearRegression()


# In[ ]:


model = mod.fit(x_train, y_train)


# In[ ]:


model.intercept_


# In[ ]:


model.coef_


# In[ ]:


y_predict = mod.predict(x_test)


# In[ ]:


r2_score(y_test, y_predict)


# In[ ]:


linear_variance  = y_predict.sum() - y_test.sum()


# In[ ]:


# Decision Tree
tree = DecisionTreeRegressor(min_samples_leaf = 0.001)
dt_model = tree.fit(x_train, y_train)
tree_y_predict = tree.predict(x_test)


# In[ ]:


r2_score(y_test, tree_y_predict)


# In[ ]:


dt_variance  = tree_y_predict.sum() - y_test.sum()


# In[ ]:


# Random Forest
rf = RandomForestRegressor(n_estimators = 100)

rf_model = rf.fit(x_train, y_train)

rf_y_predict = rf.predict(x_test)


# In[ ]:


r2_score(y_test, rf_y_predict)


# In[ ]:


rf_variance = rf_y_predict.sum() - y_test.sum()

