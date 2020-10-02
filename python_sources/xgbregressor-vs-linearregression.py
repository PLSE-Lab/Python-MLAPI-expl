#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 
import os


print(os.listdir("../input"))


# In[ ]:


data = pd.read_csv("../input/renfe.csv")
data = data.dropna()


# In[ ]:


data.head(5)


# In[ ]:


data.columns


# In[ ]:


data = data.drop("Unnamed: 0", axis=1)


# In[ ]:


labels = data["price"]


# In[ ]:


sns.distplot(labels)
plt.show()


# In[ ]:


labels = labels.values.reshape(-1, 1)


# In[ ]:


labels.shape


# In[ ]:


data = data.drop("price", axis=1)


# In[ ]:


col = ['insert_date', 'start_date', 'end_date']

for i in col:
    date = pd.to_datetime(data[i])
    data[i + ':hour'] = date.dt.hour
    data[i + ':minute'] = date.dt.minute
    data[i + ':second'] = date.dt.second
    data[i + ':weekday'] = date.dt.weekday_name
    data[i + ':day'] = date.dt.day
    data[i + ':month'] = date.dt.month
    data[i + ':year'] = date.dt.year


# In[ ]:


data = data.drop(col, axis=1)


# In[ ]:


data.shape


# In[ ]:


data = data.values


# In[ ]:


encoder = OneHotEncoder()
data = encoder.fit_transform(data)


# In[ ]:


data.shape


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)


# **XGB Regressor**

# In[ ]:


xgbr = XGBRegressor(colsample_bytree=0.4,
            gamma=0,
            learning_rate=0.07,
            max_depth=3,
            min_child_weight=1.5,
            n_estimators=1500,                                                                 
            reg_alpha=0.75,
            reg_lambda=0.45,
            subsample=0.6,
            seed=64)


# In[ ]:


xgbr.fit(x_train, y_train)


# **Linear Regression**

# In[ ]:


reg = LinearRegression()

reg.fit(x_train, y_train)


# In[ ]:


pred = xgbr.predict(x_test)
reg_pred = reg.predict(x_test)

y = y_test.reshape(y_test.shape[0], )

data_ = {'price': y,
         'XGB_predict': pred,
         'LinearRegression_predict': reg_pred.reshape(y.shape[0], )}

test = pd.DataFrame(data=data_)


# In[ ]:


test.head(10)


# In[ ]:


print("XGBRegressor score: ", xgbr.score(x_test, y_test))
print("Linear Regression score: ", reg.score(x_test, y_test))


# In[ ]:




