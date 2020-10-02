#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import cufflinks as cf
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
cf.go_offline()
from plotly import __version__
import plotly.graph_objs as go
import re
import plotly.express as px


# In[ ]:


df=pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


# see if there is any missing value
df.isnull().sum()


# In[ ]:


# drop all the missing
df.dropna(inplace=True)


# # the most popular app category

# In[ ]:


df['Category'].value_counts().head()


# In[ ]:


df['Category'].value_counts().iplot(kind='bar')


# # now let see the rating distribution

# In[ ]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['Rating'],shade=True,)


# the distribution is skewed to the left in around 4 

# In[ ]:


df["Rating"].describe()


# In[ ]:


df['Rating'].iplot('hist')


# In[ ]:


box_age = df[['Category', 'Rating']]
box_age.pivot(columns='Category', values='Rating').iplot(kind='box')


# Rating of application in each category is not realy different

# # Reviews

# change the type of data from str to int

# In[ ]:


type(df['Reviews'].iloc[0])


# In[ ]:


df['Reviews']=df['Reviews'].apply(lambda x:int(x))
print(type(df['Reviews'].iloc[0]))


# In[ ]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['Reviews'],shade=True,)
plt.xlabel("Reviews",size=15)
plt.ylabel("Frequency",size=15)
plt.title('Distribution of Reveiw',size = 20)


# In[ ]:


fig = px.scatter(df, x="Reviews", y="Rating",  log_x=False, size_max=30)
fig.show()


# In[ ]:


fig = px.scatter(df, x="Reviews", y="Rating",  log_x=True, size_max=30)
fig.show()


# # Size

# In[ ]:


df['Size'].value_counts().head(10).iplot('bar')


# In[ ]:


df['Size'].replace('Varies with device', np.nan, inplace = True )


# In[ ]:


df['Size']=(df.Size.replace(r'[kM]+$', '', regex=True).astype(float) *              df.Size.str.extract(r'[\d\.]+([KM]+)', expand=False)
            .fillna(1)
            .replace(['k','M'], [10**3, 10**6]).astype(int))
df['Size'].fillna(df.groupby('Category')['Size'].transform('mean'),inplace = True)


# In[ ]:


fig = px.density_contour(df, x="Size", y="Rating")
fig.update_traces(contours_coloring="fill", contours_showlabels = True)
fig.show()


#  we can say the size in  range 5M:25M (the yellow and the orange)
# 

# # Installs
# 

# In[ ]:


df['Installs'] = df['Installs'].apply(lambda x: x.replace(',',''))
df['Installs']  = df['Installs'].apply(lambda x:int(x.replace('+','')))


# In[ ]:


df['Installs'].value_counts()


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(df['Installs'])
plt.xticks(rotation=90)
plt.show()


# In[ ]:


df[df['Installs']>=1000000000]['App'].unique()


# we can see the most installs app are very famous

# # Type

# In[ ]:


labels = ['free', 'paid']
Free=df['Type'].value_counts()['Free']
Paid=df['Type'].value_counts()['Paid']
values = [Free,Paid]
trace = go.Pie(labels = labels, values = values)
data = [trace]
fig = go.Figure(data = data)
iplot(fig)


# Most of application in this store are free (93.1%).
# 

# # Price

# In[ ]:


#clean the data
df['Price']=df['Price'].apply(lambda x:float(x.strip('$')))


# In[ ]:


px.histogram(df[df['Price']>0]['Price'])


# In[ ]:


df[df['Price']>0]['Price'].value_counts().head(10)


# In[ ]:


df[df['Price']>0]['Price'].describe()


# # Installs - Price

# In[ ]:


px.scatter(df,y='Price',x='Installs', log_x=True, size_max=30)


#  most people insttal the free app 

# In[ ]:


df[(df['Price']>=250)&(df["Installs"]>1000)]


#  as we can see the app called(I am rich ) is very  successful 
