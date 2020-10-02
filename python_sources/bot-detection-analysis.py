#!/usr/bin/env python
# coding: utf-8

# ### BOT Detection using Unsupervised Algorithm
# 
# ### Problem Statement:- Determine the unique IP which is suspected to be BOT.

# #### Load the important Libraries to perform task

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')
from IPython.core.display import display, HTML
display(HTML("<style>.container {width:100% !important;}</style>"))


# In[ ]:


df=pd.read_csv('../input/botdetection/ibm_data.csv',index_col=0)


# In[ ]:


df.head(10)


# In[ ]:


# Let's bring some statistical insights fro data
df.describe()


# In[ ]:





# In[ ]:


# Let's check the inforamtion and type
df.info()


# In[ ]:


# check fro null values
df.isnull().sum()


# In[ ]:


import datetime
df['page_vw_ts']=pd.to_datetime(df['page_vw_ts'])


# In[ ]:


df.page_vw_ts.dt.dayofyear.head()


# In[ ]:


df['ip_addr']= df['ip_addr'].astype(str)
df['VISIT']= df['VISIT'].astype(int)
df['ENGD_VISIT']= df['ENGD_VISIT'].astype(int)
df['VIEWS']= df['VIEWS'].astype(int)
df['wk']= df['wk'].astype(int)


# In[ ]:


df['day']=df['page_vw_ts'].dt.weekday


# In[ ]:


df.head()


# In[ ]:


#Let's see the trend of year,month and day
df['year']=df.page_vw_ts.dt.year


# In[ ]:


df.head()


# In[ ]:


df['month']=df.page_vw_ts.dt.month


# In[ ]:


df.head()


# In[ ]:


df.year.value_counts().sort_index().plot()


# In[ ]:


df.month.value_counts().sort_index().plot()


# In[ ]:


df.day.value_counts().sort_index().plot()


# * as shown in above we can see the value of most of the year counts are 2019.
# * the most of the month counts are  6.
# * the most of the day counts are 1.

# In[ ]:


df.head()


# In[ ]:


df.corr()


# In[ ]:


# lets's drop wk,mth,yr as it contins most of the null values and does not play any important role
df.drop('wk',axis=1,inplace=True)


# In[ ]:


df.drop('mth',axis=1,inplace=True)
df.drop('yr',axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


# lets's drop some more columns that are irrelevant to our data means if they dont be in data set it will not affect our analysis.
df.drop('intgrtd_mngmt_name',axis=1,inplace=True)
df.drop('intgrtd_operating_team_name',axis=1,inplace=True)
df.drop('st',axis=1,inplace=True)
df.drop('sec_lvl_domn',axis=1,inplace=True)
df.drop('device_type',axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df['city'].fillna((df['city'].mode()[0]),inplace=True)


# In[ ]:


df=df.dropna()


# In[ ]:


df.isnull().sum()


# In[ ]:


sns.pairplot(df)


# * from above we can say that the VIEWS and VISITS have the linear Relationship.
# * in months most of the values are at 6.
# * in day the values are at 1.

# In[ ]:


plt.figure(figsize=[10,5])
plt.subplot(1,2,1)
sns.scatterplot(df.VIEWS,df.day,color='r')
sns.scatterplot(df.VIEWS,df.VISIT,color='g')
sns.scatterplot(df.VISIT,df.ENGD_VISIT,color='b')

plt.subplot(1,2,2)
sns.scatterplot(df.VIEWS,df.month,color='r')
sns.scatterplot(df.VIEWS,df.VISIT,color='g')
sns.scatterplot(df.VISIT,df.ENGD_VISIT,color='b')


# In[ ]:





# In[ ]:


df.pivot_table(['VISIT','VIEWS','ENGD_VISIT'],('day')).plot(kind='bar')


# In[ ]:


df.pivot_table(['VISIT','VIEWS','ENGD_VISIT'],('month')).plot(kind='bar')


# In[ ]:


data=df.pivot_table(['VISIT','VIEWS','ENGD_VISIT','day','month'],('ip_addr'),aggfunc='sum')


# In[ ]:


data


# #### 449283 Unique ip_addr with data

# In[ ]:


columns=['VISIT','VIEWS','ENGD_VISIT','day','month']
df1=pd.DataFrame(data[columns])
df1.dropna(inplace=True)


# In[ ]:


df1


# In[ ]:


# Let's scale the data fisrt
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df1.VIEWS=scaler.fit_transform(df1[['VIEWS']])
df1.VISIT=scaler.fit_transform(df1[['VISIT']])
df1.ENGD_VISIT=scaler.fit_transform(df1[['ENGD_VISIT']])
df1.day=scaler.fit_transform(df1[['day']])
df1.month=scaler.fit_transform(df1[['month']])


# In[ ]:


# using k-means to make 2 cluster groups
from sklearn.cluster import KMeans
km=KMeans(n_clusters=2)


# In[ ]:





# In[ ]:


y_pred=km.fit_predict(df1)


# In[ ]:


df1['cluster']=y_pred


# In[ ]:


# For plotting the graph of cluster
p=df1[df1.cluster==0]
q=df1[df1.cluster==1]


# In[ ]:


plt.scatter(p.VISIT,p.day,color='g')
plt.scatter(p.VISIT,p.VIEWS,color='g')
plt.scatter(p.VISIT,p.ENGD_VISIT,color='g')

plt.scatter(q.VISIT,q.day,color='b')
plt.scatter(q.VISIT,q.VIEWS,color='b')
plt.scatter(q.VISIT,q.ENGD_VISIT,color='b')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='red')


# In[ ]:


plt.scatter(p.VISIT,p.month,color='g')
plt.scatter(p.VISIT,p.VIEWS,color='g')
plt.scatter(p.VISIT,p.ENGD_VISIT,color='g')

plt.scatter(q.VISIT,q.month,color='b')
plt.scatter(q.VISIT,q.VIEWS,color='b')
plt.scatter(q.VISIT,q.ENGD_VISIT,color='b')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='red')


# * Reds are the Center point for K
# * Blues are the BOTS
# * Greens are the Humans

# In[ ]:


data['BOT']=y_pred


# In[ ]:


data[data.BOT==1]


# ### Overall 814 ip detected as a BOT.

# ### Thank YOU.

# In[ ]:




