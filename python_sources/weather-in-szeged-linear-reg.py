#!/usr/bin/env python
# coding: utf-8

# Weather in Szeged 2006-2016: Is there a relationship between humidity and temperature? What about between humidity and apparent temperature? Can you predict the apparent temperature given the humidity?

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


data_all= pd.read_csv("/kaggle/input/szeged-weather/weatherHistory.csv")


# In[ ]:


data_all.rename(columns={"Apparent Temperature (C)": "app_temp", "Humidity": "humidity"}, inplace=True)   
app_temp = data_all.app_temp
humidity = data_all.humidity


# In[ ]:


data =pd.concat([app_temp,humidity],axis=1, ignore_index=True)
data.rename(columns={0: "app_temp", 1: "humidity"}, inplace=True)


# In[ ]:


data.head()


# In[ ]:


data.isna().sum()


# In[ ]:


data.corr()


# In[ ]:


data.info()


# In[ ]:


data.describe().T


# In[ ]:


data.count()


# In[ ]:


x=data[["humidity"]]
y=data["app_temp"]


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state= 0)


# In[ ]:


model = LinearRegression().fit(x_train,y_train)  # ornek modeli tanimliyoruz


# In[ ]:


y_pred = model.predict(x_test)
y_pred


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
r2_score(y_test, y_pred)


# In[ ]:


model.predict([[0.89]])


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
aray = np.arange(len(y_test))
plt.plot(aray, y_pred, color="red" )  
plt.plot(aray, y_test, color="blue",alpha=0.5)

plt.show();


# In[ ]:


plt.plot(x_test, y_test,  color='black')
plt.plot(y_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


# In[ ]:




