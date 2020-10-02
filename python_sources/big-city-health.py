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

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))
big = pd.read_csv("../input/Big_Cities_Health_Data_Inventory.csv")


# In[ ]:


big.columns


# In[ ]:


big.shape


# 
# **Observations**
# 
# There are 13512 observations and 11 columns for Big City Health dataset.
# 

# In[ ]:


big.info()


# In[ ]:


big.head(10).T


# In[ ]:


big.tail(10)


# **Number of Null Values in each column**

# In[ ]:


big.isna().sum()


# **Basic Statistics of Dataframe**

# In[ ]:


big.describe(include='all')


# **Remove all Null values in Value**

# In[ ]:


big.dropna(subset=["Value"],inplace=True)


# **Replace the Null values in Value with median**

# In[ ]:


big.Value.fillna(big.Value.median(),inplace=True)


# **Check for Duplicated Data**

# In[ ]:


big.duplicated().sum()


# In[ ]:


big = big.drop_duplicates()


# In[ ]:


big.duplicated().sum()


# **Conversion of Float to Int for "Value" Column**

# In[ ]:


big.loc[big['Value'].notnull(),'Value'].apply(int)


# **Replace the Null Values with Mode**

# In[ ]:


for column in ['Source','BCHC Requested Methodology']:
    big[column].fillna(big[column].mode()[0], inplace=True)


# In[ ]:


big.isna().sum()


# In[ ]:


big.drop(columns=['Methods','Notes'],inplace=True)


# In[ ]:


big.head(2)


# In[ ]:


big['Indicator Category'].value_counts()


# In[ ]:


groupvalues=big.groupby('Indicator Category').sum().reset_index()


# In[ ]:


groupvalues.head()


# In[ ]:


plt.figure(figsize=(40,10)) 
sns.set(style="whitegrid")
groupvalues=big.groupby('Indicator Category').sum().reset_index()
g = sns.barplot(groupvalues['Indicator Category'],groupvalues['Value'])
for index, row in groupvalues.iterrows():
    g.text(row.name,row.Value, round(row.Value,2), color='black', ha="center")
    g.set_xlabel("Indicator Category", fontsize=25)
plt.show()


# In[ ]:


plt.figure(figsize=(40,10)) 
sns.set(style="whitegrid")
groupvalues=big.groupby('Place').sum().reset_index()
g = sns.barplot(groupvalues['Place'],groupvalues['Value'])
for index, row in groupvalues.iterrows():
    g.text(row.name,row.Value, round(row.Value,2), color='black', ha="center")
    g.set_xlabel("Place", fontsize=25)
plt.show()


# In[ ]:


plt.figure(figsize = (40,25))
big.Place.value_counts().plot(kind="pie")


# In[ ]:


import numpy
plt.figure(figsize = (10,5))
labels = 'Male', 'Female', 'Both'
sizes = numpy.array([1680, 2423, 9409])
colors = ['yellowgreen', 'violet', 'yellow']

p, tx, autotexts = plt.pie(sizes, labels=labels, colors=colors,
        autopct="", shadow=True)

for i, a in enumerate(autotexts):
    a.set_text("{}".format(sizes[i]))

plt.axis('equal')
plt.show()


# In[ ]:


plt.figure(figsize=(14, 7))
sns.scatterplot(x=big['Year'], y=big['Value'],hue=big['Gender'])


# In[ ]:


big.isna().sum()


# In[ ]:


big_encoded = pd.get_dummies(big)


# In[ ]:


X = big_encoded.drop(columns=['Value'])
y = big_encoded['Value']


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 101)
model = LinearRegression()
model.fit(X_train,y_train)


# In[ ]:


model.coef_


# In[ ]:


model.intercept_


# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[ ]:


train_predict = model.predict(X_train)

mae_train = mean_absolute_error(y_train,train_predict)

mse_train = mean_squared_error(y_train,train_predict)

rmse_train = np.sqrt(mse_train)

r2_train = r2_score(y_train,train_predict)

mape_train = mean_absolute_percentage_error(y_train,train_predict)


# In[ ]:


test_predict = model.predict(X_test)

mae_test = mean_absolute_error(test_predict,y_test)

mse_test = mean_squared_error(test_predict,y_test)

rmse_test = np.sqrt(mean_squared_error(test_predict,y_test))

r2_test = r2_score(y_test,test_predict)

mape_test = mean_absolute_percentage_error(y_test,test_predict)


# In[ ]:


print('TRAIN: Mean Absolute Error(MAE): ',mae_train)
print('TRAIN: Mean Squared Error(MSE):',mse_train)
print('TRAIN: Root Mean Squared Error(RMSE):',rmse_train)
print('TRAIN: R square value:',r2_train)
print('TRAIN: Mean Absolute Percentage Error: ',mape_train)
print('TEST: Mean Absolute Error(MAE): ',mae_test)
print('TEST: Mean Squared Error(MSE):',mse_test)
print('TEST: Root Mean Squared Error(RMSE):',rmse_test)
print('TEST: R square value:',r2_test)
print('TEST: Mean Absolute Percentage Error: ',mape_test)

