#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


data_set=pd.read_csv('../input/ecommerce-behavior-data-from-multi-category-store/2019-Nov.csv',nrows=100000)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x=data_set.iloc[:,0:8]
y=data_set.iloc[:,8]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size =0.90,random_state=0)


# In[ ]:


x_test.info()


# In[ ]:


df1=pd.DataFrame(x_train)


# In[ ]:


df2=pd.DataFrame(y_train)


# In[ ]:


df= df1.join(df2)


# In[ ]:


df.head(10)


# In[ ]:


df['event_type'].value_counts()


# In[ ]:


df.loc[df['category_id']==2090971680783466556].sort_values(by=['event_time'])


# In[ ]:


df= df.reset_index()


# In[ ]:


df.drop(['index'],axis=1) 


# In[ ]:


df.info()


# In[ ]:


df.event_time = pd.to_datetime(df.event_time, utc=True)

df['day']=df['event_time'].dt.day
df['hour']=df['event_time'].dt.hour
df['minute']=df['event_time'].dt.minute
df['seconds']=df['event_time'].dt.second


# In[ ]:


df.drop(['index'],axis=1,inplace=True) 


# In[ ]:


df.drop(['event_time'],axis=1,inplace=True) 


# In[ ]:


df['event_type'].value_counts()


# In[ ]:



import matplotlib.pyplot as plt
labels = ['view', 'purchase','cart']
size = df['event_type'].value_counts()
colors = ['lightgreen', 'orange','blue']
explode = [0, 0.1,0.1]

plt.rcParams['figure.figsize'] = (8, 8)
plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
plt.title('Event_Type', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()


# In[ ]:


df.drop(['category_code'],axis=1,inplace=True) 


# In[ ]:


df.drop(['day'],axis=1,inplace=True) 


# In[ ]:


df.loc[df.category_id==2053013553559896355].sort_values(by='hour')


# In[ ]:


df


# In[ ]:


df['event_type'] = df['event_type'].astype('category').cat.codes
#purchase=1 View=2 Cart=0


# In[ ]:


df['event_type'] = df['event_type'].astype('int')


# In[ ]:


df.info()


# In[ ]:



df['brand'].value_counts().head(50).plot.bar(figsize = (18, 7))
plt.title('Top Brand', fontsize = 20)
plt.xlabel('Names of Brand')
plt.ylabel('Count')
plt.show()


# In[ ]:


df['event_type'].groupby(df['brand']).agg('sum').sort_values(ascending = False).head(20).plot.bar(figsize = (15, 7), color = 'lightblue')
plt.title('Top 20 Countries Sales wise', fontsize = 20)
plt.xlabel('Names of Countries')
plt.ylabel('Sales')
plt.show()


# In[ ]:


df.head(5)


# In[ ]:


df['user_session'] = df['user_session'].astype('category').cat.codes.astype(
    'category')
df['user_id'] = df['user_id'].astype('category').cat.codes.astype('category')
df['category_id'] = df['category_id'].astype('category').cat.codes.astype('category')
df['product_id'] = df['product_id'].astype('category').cat.codes.astype('category')


# In[ ]:


df


# In[ ]:


df.info()


# In[ ]:




