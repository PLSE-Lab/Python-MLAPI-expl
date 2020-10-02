#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data=pd.read_csv('../input/cars.csv')
data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


sns.set_style('whitegrid')
sns.lmplot('mpg',' cylinders',data=data,hue=' brand',fit_reg=False)


# In[ ]:


data.columns


# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


kmc=KMeans(n_clusters=3)


# In[ ]:


data.info()


# In[ ]:


data[data[' cubicinches']==" "]


# In[ ]:


df=data[data[' cubicinches']!=" "]


# In[ ]:


df[' cubicinches']=df[' cubicinches'].astype(float)


# In[ ]:


df[df[' weightlbs']==' ']


# In[ ]:


df=df[df[' weightlbs']!=' ']


# In[ ]:


df[' weightlbs']=df[' weightlbs'].astype(float)


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


sns.pairplot(df,hue=' brand')


# In[ ]:


kmc.fit(df.drop(' brand',axis=1))


# In[ ]:


kmc.cluster_centers_


# In[ ]:


kmc.labels_


# In[ ]:


def converter(x):
    if x==' US.':
        return 2
    elif x==' Europe.':
        return 0
    elif x==' Japan.':
        return 1
    else:
        return 2


# In[ ]:


df['cluster']=df[' brand'].apply(converter)


# In[ ]:


df.head()


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['cluster'],kmc.labels_))
print(classification_report(df['cluster'],kmc.labels_))


# In[ ]:


plt.figure(figsize=(12,6))
sns.lmplot(x=' weightlbs',y=' cubicinches',data=df,hue='cluster',fit_reg=False)


# In[ ]:


df['pred_clusters']=kmc.labels_


# In[ ]:


df.head()


# In[ ]:


plt.figure(figsize=(12,6))
sns.lmplot(x=' weightlbs',y=' cubicinches',data=df,hue='pred_clusters',fit_reg=False)


# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(14,6))
ax1.set_title('K Means')
ax1.scatter(df[' weightlbs'],df[' cubicinches'],c=df['pred_clusters'],cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(df[' weightlbs'],df[' cubicinches'],c=df['cluster'],cmap='rainbow')

