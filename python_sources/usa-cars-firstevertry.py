#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns 
import scipy 
import sklearn 
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans


# In[ ]:


df = pd.read_csv('../input/usa-cers-dataset/USA_cars_datasets.csv')


# In[ ]:





# In[ ]:


df.head(2)


# In[ ]:


df.describe() 


# In[ ]:


df.mean()


# In[ ]:


df.info ()


# In[ ]:


df.head()


# In[ ]:


X = df.drop('country', axis = 1 )


# In[ ]:


X.head()


# In[ ]:


X = X[[ 'price', 'brand',  'model', 'year', 'title_status', 'mileage','color','vin','lot','state', 'condition', ]]


# In[ ]:


X.head(20)


# In[ ]:


X.shape


# In[ ]:


X = X.fillna('')


# In[ ]:


X_transformed=X[X.year >=2006]


# In[ ]:


#sns.boxplot(X.year)


# In[ ]:


X_transformed.year.hist()


# In[ ]:


sns.boxplot(X_transformed.year)


# In[ ]:


X.year.unique()


# In[ ]:


X_transformed.year.unique()


# In[ ]:


X.title_status.unique()


# In[ ]:


X.brand.unique()


# In[ ]:


sns.pairplot(X_transformed[['price', 'brand','mileage','year']], hue = 'year')


# In[ ]:


price = X.groupby('brand')['price'].max().reset_index()
price  = price.sort_values(by="price")
price = price.tail(15)
fig = px.pie(price,
             values="price",
             names="brand",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.005, textinfo="percent+label")
fig.show()


# In[ ]:


year = px.data.iris()
fig = px.scatter(df, x="year", y="price", )
fig.show()


# In[ ]:


#df3 = pd.read_csv('USA_cars_datasets.csv')


# In[ ]:


#X[["F"]] = X3[["D", "E"]].astype(int)


# In[ ]:


#X['condition'] = X['condition'].astype(str)


# In[ ]:


X_3=X[X.year == 2006]


# In[ ]:


X_transformed= px.data.tips()
fig = px.sunburst(df, path=['year', 'title_status', 'brand'], values='price')
fig.show()


# In[ ]:


X.mean()


# In[ ]:


#kmeans = KMeans(n_clusters = 6, random state=1)f.fit(X.mean())


# In[ ]:


sns.heatmap(X.corr())


# In[ ]:


df['brand'].value_counts()[:30]


# In[ ]:


df['brand'].value_counts()[:30].plot(kind='barh')
    


# In[ ]:


X['state'].value_counts()[:50]


# In[ ]:


X['state'].value_counts()[:50].plot(kind='bar',)


# In[ ]:


X['title_status'].value_counts()[:2]


# In[ ]:


labels = 'clean vehicle', 'salvage insurance'
sizes = [2336,163]
explode = (0, 0.5) 

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=180)
ax1.axis('on')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
#percentage of total vehicles salvage/clean


# In[ ]:


X['year'].value_counts()[:30]


# In[ ]:


year = px.data.iris()
fig = px.scatter(df, x="year", y="brand")
fig.show()


# In[ ]:


X['year'].value_counts()[:30].plot(kind='bar',)


# In[ ]:


year = px.data.iris()
fig = px.scatter(X, x="year", y="state")
fig.show()


# 

# In[ ]:




