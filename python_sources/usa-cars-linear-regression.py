#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats 
from  sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[ ]:


data = pd.read_csv("/kaggle/input/vehicle-dataset-from-cardekho/car data.csv")


# In[ ]:


data


# In[ ]:


data.info()


# # Original Distributions

# In[ ]:


check_norm=data[['Present_Price','Kms_Driven']]


# In[ ]:


for x in check_norm:
    plt.figure(figsize=(8,3))
    sns.distplot(check_norm[x])


# # Log-Normal Distributions

# In[ ]:


for x in check_norm:
    logs=np.log(check_norm[x])
    plt.figure(figsize=(13,4))
    plt.subplot(1,2,1)
    plt.title(x)
    sns.distplot(logs)
    plt.subplot(1,2,2)
    stats.probplot(logs,dist='norm',plot=plt)
    plt.show()
    


# # We can take the log-normal distribution But before this i want to remove outliers from the features

# # Removing Outliers

# In[ ]:


fig,ax=plt.subplots(figsize=(15,10))
sns.boxplot(data=check_norm,ax=ax, fliersize=3)


# In[ ]:


data_cleaned=data[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]


# In[ ]:


q=data_cleaned['Kms_Driven'].quantile(0.97)
data_cleaned=data_cleaned[data_cleaned['Kms_Driven']<q]
q=data_cleaned['Present_Price'].quantile(0.99)
data_cleaned=data_cleaned[data_cleaned['Present_Price']<q]


# In[ ]:


data_cleaned.info()


# In[ ]:


for x in data_cleaned[['Present_Price','Kms_Driven']]:
    plt.figure(figsize=(9,3))
    sns.distplot(data_cleaned[x])


# In[ ]:


for x in data_cleaned[['Present_Price','Kms_Driven']]:
    logs=np.log(data_cleaned[x])
    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    sns.distplot(logs)
    plt.subplot(1,2,2)
    stats.probplot(logs,dist='norm',plot=plt)
    plt.show()


# In[ ]:


fig,ax=plt.subplots(figsize=(15,10))
sns.boxplot(data=data_cleaned,ax=ax, fliersize=3)


# # Now we have to fix the distribution of the data By taking log-transformation

# In[ ]:


for x in data_cleaned[['Present_Price','Kms_Driven']]:
    logss_cleaned=np.log(data_cleaned[['Present_Price','Kms_Driven']][x])
    plt.figure(figsize=(15,3))
    plt.subplot(1,2,1)
    sns.distplot(logss_cleaned)
    plt.subplot(1,2,2)
    stats.probplot(logss_cleaned,dist='norm',plot=plt)
    plt.show()
    


# In[ ]:


data_cleaned['Present_Price']=np.log(data_cleaned['Present_Price'])
data_cleaned['Kms_Driven']=np.log(data_cleaned['Kms_Driven'])


# In[ ]:


data_cleaned


# # Handling-Categorical variables

# In[ ]:


data_cleaned=pd.get_dummies(data_cleaned,drop_first=True)


# In[ ]:


data_cleaned=data_cleaned.drop(columns="Fuel_Type_Diesel")


# In[ ]:


data_cleaned


# In[ ]:


x=data_cleaned.drop(columns='Selling_Price')
y=data_cleaned['Selling_Price']


# # Now scaling the features

# In[ ]:


scaler=StandardScaler()
X_scaled=scaler.fit_transform(x)


# In[ ]:


X_scaled


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.25,random_state=42)


# In[ ]:


model=LinearRegression()
model.fit(X_train,y_train)


# In[ ]:


ytrain_predict=model.predict(X_train)
ytrain_predict


# In[ ]:


print("The R square is equal to {}".format(r2_score(y_train,ytrain_predict)))


# In[ ]:


ytest_predict=model.predict(X_test)
ytest_predict


# In[ ]:


print("The R square is equal to {}".format(r2_score(y_test,ytest_predict)))


# In[ ]:


print("The r square value on test data is {}".format(model.score(X_test,y_test)))
print("The r square value on train data is  {}".format(model.score(X_train,y_train)))


# In[ ]:




