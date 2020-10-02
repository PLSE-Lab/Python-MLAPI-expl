#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()


# In[ ]:


df=pd.read_csv("../input/car-price-prediction/CarPrice_Assignment.csv")
df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.columns


# In[ ]:


df_2=df.drop(['carwidth'],axis=1)


# In[ ]:


df_3=df_2.drop(['wheelbase'],axis=1)


# In[ ]:


df_4=df_3.drop(['carlength'],axis=1)


# In[ ]:


df_5=df_4.drop(['carheight'],axis=1)


# In[ ]:


df_6=df_5.drop(['highwaympg'],axis=1)


# In[ ]:


df_7=df_6.drop(['boreratio'],axis=1)


# In[ ]:


df_8=df_7.drop(['curbweight'],axis=1)


# In[ ]:


df_9=df_8.drop(['peakrpm'],axis=1)


# In[ ]:


df_10=df_9.drop(['stroke'],axis=1)


# In[ ]:


df_11=df_10.drop(['enginesize'],axis=1)


# In[ ]:


df_12=df_11.drop(['citympg'],axis=1)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables=df_11[['car_ID', 'symboling',
       'compressionratio', 'horsepower']]
vif=pd.DataFrame()
vif['VIF']=[variance_inflation_factor(variables.values,i)for i in range(variables.shape[1])]
vif['Features']=variables.columns
vif


# In[ ]:


data_w_d=pd.get_dummies(df_12,drop_first=True)


# In[ ]:


data_w_d.columns


# In[ ]:


f,(x1,x2,x3)=plt.subplots(1,3,sharey=True,figsize=(15,3))
x1.scatter(data_w_d['symboling'],data_w_d['price'])
x1.set_title('Price and symbolling')
x2.scatter(data_w_d['compressionratio'],data_w_d['price'])
x2.set_title('Price and compressionratio')
x3.scatter(data_w_d['horsepower'],data_w_d['price'])
x3.set_title('Price and horsepower')


# In[ ]:


log_price=np.log(data_w_d['price'])
data_w_d['log_price']=log_price


# In[ ]:


f,(x1,x2,x3)=plt.subplots(1,3,sharey=True,figsize=(15,3))
x1.scatter(data_w_d['symboling'],data_w_d['log_price'])
x1.set_title('Price and symbolling')
x2.scatter(data_w_d['compressionratio'],data_w_d['log_price'])
x2.set_title('Price and compressionratio')
x3.scatter(data_w_d['horsepower'],data_w_d['log_price'])
x3.set_title('Price and horsepower')


# In[ ]:


targets=data_w_d['log_price']
inputs=data_w_d.drop(['log_price'],axis=1)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(inputs)
scaled_inputs=scaler.transform(inputs)


# In[ ]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(scaled_inputs,targets)
reg.score(scaled_inputs,targets)


# In[ ]:


y_hat=reg.predict(scaled_inputs)


# In[ ]:


plt.scatter(targets,y_hat)


# In[ ]:


df_pf=pd.DataFrame()
df_pf['Targets']=np.exp(targets)
df_pf['Predictions']=np.exp(y_hat)
df_pf['Difference']=df_pf['Targets']-df_pf['Predictions']
df_pf['Residuls']=np.absolute(df_pf['Difference']/df_pf['Targets'])

df_pf


# In[ ]:




