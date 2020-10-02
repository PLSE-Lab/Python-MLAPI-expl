#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import numpy as np 
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:



cars = pd.read_csv("../input/carsdata/cars.csv", na_values =' ')
cars.head()


# In[ ]:


cars.columns =['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60',
       'year', 'brand']


# In[ ]:


cars =cars.dropna()
cars['cubicinches'] = cars['cubicinches'].astype(int)
cars['weightlbs'] = cars['weightlbs'].astype(int)


# In[ ]:


cars.columns = ['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60',
       'year', 'brand']


# In[ ]:


cars.columns


# In[ ]:


X = cars.iloc[:,:7]


# In[ ]:


X.head()


# In[ ]:


X.describe()


# In[ ]:


X_array = X.values  
X_array


# In[ ]:


X.head()


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X_array)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('metodo cotuvelo - ebow method')
plt.xlabel('numero de clusters')
plt.ylabel('WCSS')
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=10,random_state=0)
kmeans.fit_predict(X_array)


# In[ ]:


kmeans = KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=10,random_state=0)
cars['clusters'] = kmeans.fit_predict(X_array)
cars.head()


# In[ ]:


kmeans = KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=10,random_state=0)
cars['clusters'] = kmeans.fit_predict(X_array)
cars.head()
cars.groupby("clusters").agg('mean').plot.bar(figsize=(10,7.5))
plt.title("gastos por clusteer")


# In[ ]:




