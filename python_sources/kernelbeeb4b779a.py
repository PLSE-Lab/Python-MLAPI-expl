#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split


# In[ ]:


data=pd.read_csv("C:\\Users\\Akbar Ali\\Downloads\\ctw.csv")


# In[ ]:


print(data.isna().sum())


# In[ ]:


d1=data.fillna(method='ffill')
d1.shape


# In[ ]:


print(d1.isna().sum())


# In[ ]:


d1.corr()["Population"]


# In[ ]:


d1.corr()


# In[ ]:


d1=d1.rename(columns={"Area (sq. mi.)":"Area","GDP ($ per capita)":"GDP"})


# In[ ]:


plt.scatter(d1.Area/1000000,d1.Population/10000000)


# In[ ]:


d2=d1.drop(columns=["Country","Region","Pop. Density (per sq. mi.)","Coastline (coast/area ratio)","Net migration","Infant mortality (per 1000 births)","GDP","Literacy (%)","Phones (per 1000)","Arable (%)","Crops (%)","Other (%)","Climate","Birthrate","Deathrate","Agriculture","Industry","Service"])
d2.head()


# In[ ]:


x1=d2.Population.values/10000000
y1=d2.Area.values/1000000
plt.title("Area VS Population")
plt.xlabel("Area")
plt.ylabel("Population")
plt.scatter(x1,y1)


# In[ ]:


plt.subplot(121)
plt.boxplot(d2.Area)
plt.subplot(122)
plt.boxplot(d2.Population)
plt.show()


# In[ ]:


d3=d2[d2.Area<d2.Area.quantile(0.4)]
d3.shape


# In[ ]:


plt.subplot(121)
plt.boxplot(d3.Area)
plt.show()


# In[ ]:


X=d3.Population.values
X=X.reshape(-1,1)
y=d3.Area.values
y=y.reshape(-1,1)


# In[ ]:


area_train,area_test,population_train,population_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[ ]:


lm=linear_model.LinearRegression()
lm.fit(area_train,population_train)
test_population_pred=lm.predict(area_test)
train_population_pred=lm.predict(area_train)


# In[ ]:


e=mse(population_test,test_population_pred)
se=np.sqrt(e)
se


# In[ ]:


plt.figure(figsize=(10,5))
plt.subplot(121)
plt.scatter(area_train,population_train) 
plt.plot(area_train,train_population_pred,'b')
plt.subplot(122)
plt.scatter(area_test,population_test)
plt.plot(area_test,test_population_pred,'b')
plt.show()


# In[ ]:


plt.scatter(area_train,population_train)
plt.plot(area_test,test_population_pred,'r')


# In[ ]:


plt.scatter(area_train,population_train)
plt.plot(area_train,train_population_pred,'r')

