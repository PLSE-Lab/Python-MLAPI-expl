#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import accuracy_score ,mean_squared_error ,r2_score 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#data
gender_submission = pd.read_csv("/kaggle/input/house-price-predict/sampleSubmission.csv")
test = pd.read_csv("/kaggle/input/house-price-predict/test_data.csv")
train = pd.read_csv("/kaggle/input/house-price-predict/train_data.csv")
pr_test=pd.read_csv("/kaggle/input/house-price-predict/test_predict.csv")


# In[ ]:


print(train.head())
print(test.head())
print(gender_submission.head())


# In[ ]:


print(train.info())
print(test.info())


# In[ ]:



corr = train.corr()
plt.subplots(figsize=(15,12))
sns.heatmap(corr, vmax=0.9, cmap="Blues", square=True)


# In[ ]:


plt.figure(figsize=(10, 10))
corrMatrix=train[["total_rooms","population","median_income","median_house_value"]].corr()
sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='viridis',linecolor="white")
plt.title('Correlation between features');


# In[ ]:


train.plot(x='total_rooms', y='median_house_value', style='o')  
plt.title('total_rooms vs "median_house_value')  
plt.xlabel('total_rooms')  
plt.ylabel('"median_house_value')  
plt.show()


# In[ ]:


train.plot(x='population', y='median_house_value', style='o')  
plt.title('population vs "median_house_value')  
plt.xlabel('population')  
plt.ylabel('"median_house_value')  
plt.show()


# In[ ]:


train.plot(x='median_income', y='median_house_value', style='o')  
plt.title('median_income vs "median_house_value')  
plt.xlabel('median_income')  
plt.ylabel('"median_house_value')  
plt.show()


# In[ ]:


Y_test=test['median_house_value'].values
test.drop(['median_house_value'],axis=1,inplace=True)
Y_train=train['median_house_value'].values
train.drop(['median_house_value'],axis=1,inplace=True)


# In[ ]:



X,Y = make_regression(n_features=3,random_state=0,shuffle=False)
reg = RandomForestRegressor(max_depth =3,random_state=0)
reg.fit(X,Y)
pred = reg.predict(train)
print('score train :',(r2_score(Y_train,pred)))
pred=reg.predict(test)
print('score test :',(r2_score(Y_test,pred)))


# In[ ]:


pr_test.info()


# In[ ]:


Id =pr_test.Id 
pr_test.drop(['Id'],axis=1,inplace=True)


# In[ ]:



print(pr_test.info())
print(gender_submission.info())


# In[ ]:


#X,Y = make_regression(n_features=3,random_state=0,shuffle=False)
reg = RandomForestRegressor(n_estimators=1000,max_depth =7,random_state=0)
reg.fit(train,Y_train)
pred=reg.predict(test)
print('score test :',(r2_score(Y_test,pred)))
print('mean_squared_error :',mean_squared_error(Y_test,pred))
print('RMSE :',np.sqrt(mean_squared_error(Y_test,pred)))


# In[ ]:


pred=reg.predict(pr_test)
print(pred)
data = []
c=0
for i in range(len(pred)):
    data.append([Id[i],pred[i]])
    
gender_subm= pd.DataFrame(data, columns = ['Id','Prediction'])     
print(gender_subm)  
    
gender_subm.to_csv("gender_submission.csv", index=False)

