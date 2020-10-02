#!/usr/bin/env python
# coding: utf-8

# # Topic:Project In Bike Rental Demand

# In[ ]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statistics as stat
from sklearn.model_selection import train_test_split


# In[ ]:


pwd


# In[ ]:


cd  E:\shivam imp\samester 2\project\analysis


# In[ ]:


df=pd.read_csv("train.csv")


# In[ ]:


df.head(5)


# In[ ]:


p1=df['casual']+df['registered']


# In[ ]:


d1=df['datetime']
n1=len(d1)


# In[ ]:


s1=[];s2=[];s3=[]
date=[];time=[]
for i in range(n1):
    s1=d1[i]
    s2=s1[0:10]
    s3=s1[11:18]
    date=np.append(date,s2)
    time=np.append(time,s3)


# In[ ]:


df_date=pd.DataFrame(date)
df_date


# In[ ]:


df_time=pd.DataFrame(time)
df_time


# # Exploratory Data Analysis

# In[ ]:


for col in df.columns:
    print(col)


# In[ ]:


df.dtypes


# In[ ]:


df.describe()


# In[ ]:


print(df['holiday'].value_counts()/len(df['holiday']))
r=pd.get_dummies(df['holiday'])
r1=(sum(r[0]),sum(r[1]))
plt.pie(r1,labels=["holiday","non_holiday"],shadow=True,explode=(.25,.25))


# In[ ]:


print(df['season'].value_counts()/len(df['season']))
r2=pd.get_dummies(df['season'])
r3=(sum(r2[1]),sum(r2[2]),sum(r2[3]),sum(r2[4]))
plt.pie(r3,labels=["spring","summer","rainy","winter"],shadow=True,explode=(.10,.10,.10,.10))


# In[ ]:


print(df['workingday'].value_counts()/len(df['workingday']))
r4=pd.get_dummies(df['workingday'])
r5=(sum(r4[0]),sum(r4[1]))
plt.pie(r5,labels=["workingday","non_workingday"],shadow=True,explode=(.05,.05))


# # Regression Analysis

# In[ ]:


x=df[["holiday","workingday","weather","temp","humidity","windspeed"]]
y=df[["count"]]


# In[ ]:


from sklearn.preprocessing import StandardScaler
stc=StandardScaler()


# In[ ]:


x=stc.fit_transform(x)
y=stc.fit_transform(y)


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20)


# In[ ]:


from sklearn.linear_model import LinearRegression
lreg=LinearRegression()
lreg.fit(x_train,y_train)


# In[ ]:


plt.plot(lreg.predict(x_train)[0:100])
plt.plot(y_train[0:100])


# In[ ]:


count=lreg.predict(x_train)
print(count)


# In[ ]:


sd=np.std(df["count"].values)
m=np.mean(df["count"].values)


# In[ ]:


a=[]
b=[]
for i in count:
    b=(i*sd)+m
    a=np.append(a,b)
print(a)


# In[ ]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error


# In[ ]:


mse(a,y_train)


# In[ ]:


n=24
pi=a
a1=y_train


# In[ ]:


#Root Mean Squared Logarithmic Error (RMSLE)
def RMSLE(num,y_pri,y_act):
    ms_error=np.sqrt((np.nansum((np.log(y_pri+1)-np.log(y_act+1))**2))/n)
    return ms_error


# In[ ]:


RMSLE(n,pi,a1)


# In[ ]:





# In[ ]:




