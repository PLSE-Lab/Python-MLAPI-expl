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


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats
from scipy.stats import norm, skew
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import warnings

Atrain= pd.read_csv("../input/train.csv")
Btest= pd.read_csv("../input/test.csv")
print ("train data shape :", Atrain.shape)
print ("test data shape :", Btest.shape)
Atrain.head(5)




# In[2]:


Btest.head(5)


# In[3]:


Atrain['SalePrice'].describe()


# In[4]:


sns.distplot(Atrain['SalePrice']);


# In[5]:


print("Skewness: %f" % Atrain['SalePrice'].skew())
print("Kurtosis: %f" % Atrain['SalePrice'].kurt())


# In[6]:


fig, ax = plt.subplots()
ax.scatter(x = Atrain['GrLivArea'], y = Atrain['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# In[7]:


sns.distplot(Atrain['SalePrice'] , fit=norm);
(mu, sigma) = norm.fit(Atrain['SalePrice'])
print( '\n mu= %.2f and sigma= %.2f\n'% (mu, sigma))
plt.ylabel('Frequency')
plt.title('SalePrice distribution')


# In[8]:


fig = plt.figure()
res = stats.probplot(Atrain['SalePrice'], plot=plt)
plt.show()


# In[10]:


numeric_features=Atrain.select_dtypes(include=[np.number])
corr=numeric_features.corr()
print (corr['SalePrice'].sort_values (ascending=False)[:5], '\n' )
print (corr['SalePrice'].sort_values (ascending=False)[-5:], '\n' )


# In[11]:


Atrain= Atrain[Atrain['GarageArea']<1200]
plt.scatter(x=Atrain['GarageArea'], y=np.log(Atrain.SalePrice))
plt.xlim(-200,1600)
plt.ylabel ('SalePrice')
plt.xlabel("GarageArea")
plt.show()


# In[12]:


x = pd.DataFrame(Atrain.isnull().sum().sort_values(ascending =False)[:25])
x.column = ['null count']
x.index.name = 'Features'
print (x)


# In[13]:


categoricals = Atrain.select_dtypes(exclude=[np.number])
print(categoricals.describe())


# In[14]:


print ("original : \n")
print (Atrain.Street.value_counts(), "\n")

print ("\n")

Atrain ['enc_street']= pd.get_dummies(Atrain.Street, drop_first=True)
Btest ['enc_street']= pd.get_dummies(Atrain.Street, drop_first=True)
print ("encoded : \n")
print (Atrain.enc_street.value_counts(), "\n")


# In[15]:


data= Atrain.select_dtypes(include=[np.number]).interpolate().dropna()
print(sum(data.isnull().sum()!=0))
y=np.log(Atrain.SalePrice)
x=data.drop(['SalePrice', 'Id'], axis=1)


# In[17]:


x_Atrain, x_Btest, y_Atrain, y_Btest= train_test_split(x,y,random_state=12, test_size=.33)
lr=LinearRegression()
model=lr.fit(x_Atrain,y_Atrain)


# In[18]:


print ('R^2 \n',model.score(x_Btest,y_Btest))
prediction=model.predict(x_Btest)
print ('RMSE \n',mean_squared_error(y_Btest,prediction)) 


# In[19]:


actual_value=y_Btest
plt.scatter(prediction, actual_value, alpha=.75)
plt.xlabel("Prediction_price")
plt.ylabel("Actual_price")
plt.title("Linear_Regression_Model")
plt.show()


# In[20]:


submission=pd.DataFrame()
submission['Id']=Btest.Id
feats=Btest.select_dtypes(include=[np.number]).drop(['Id'],axis=1).interpolate()
prediction=model.predict(feats)
print("original_prediction:", prediction[:10], "\n")


# In[21]:


final_prediction=np.exp(prediction)
print ("final_Prediction :", final_prediction[:10])


# In[22]:


submission['SalePrice']=final_prediction
print(submission.head())


# In[23]:


submission.to_csv("submissionnew.csv", index=False)


# In[ ]:





# In[ ]:





# 

# 
