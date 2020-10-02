#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

d = pd.read_csv('../input/ks-projects-201801.csv',encoding='latin1')


# In[ ]:


df = pd.DataFrame(d)


# In[ ]:


df.head()


# In[ ]:


#mapping the state of kick starter project to 5 classes
df['state']=df['state'].map({'failed':0,'successful':1,'canceled':2,'live':3,'undefined':4,'suspended':5})


# In[ ]:


sns.countplot(x='state',data=df)


# In[ ]:


# as classes 3,4,5 are very low in count model might not train well so i am droping class 2,3,4,5. 
# Analysing only sucess & failure projects 
dfx=df[df['state']<2]
sns.countplot(x='state',data=dfx)


# In[ ]:


dfx.head()


# In[ ]:


sns.jointplot(x=dfx['state'],y=dfx['backers'],data=dfx)


# In[ ]:


sns.barplot(x='state',y='backers',data=dfx,hue='state')


# In[ ]:


sns.barplot(x='state',y='backers',data=dfx,hue='state',estimator=np.std)


# In[ ]:


sns.barplot(x='state',y='pledged',data=dfx,hue='state')


# In[ ]:


sns.barplot(x='state',y='goal',data=dfx,hue='state')


# In[ ]:


sns.barplot(x='state',y='goal',data=dfx,hue='state',estimator=np.std)


# In[ ]:


sns.countplot(x='state',data=dfx)
ax=sns.countplot(x='state',data=dfx)
ax


# In[ ]:


sns.boxplot(x='state',y='goal',data=dfx)


# In[ ]:


sns.boxplot(x='state',y='pledged',data=dfx,hue='state')


# In[ ]:


dfx.head()


# In[ ]:


sns.stripplot(x='state',y='backers',data=dfx,jitter=True)


# In[ ]:


sns.stripplot(x='state',y='goal',data=dfx,jitter=True)


# In[ ]:


sns.stripplot(x='state',y='pledged',data=dfx,jitter=True)


# In[ ]:


sns.stripplot(x='state',y='usd_goal_real',data=dfx,jitter=True)


# In[ ]:


sns.stripplot(x='state',y='usd_pledged_real',data=dfx,jitter=True)


# In[ ]:


heatt=dfx.corr()


# In[ ]:


sns.heatmap(heatt,annot=True)


# In[ ]:


cco=dfx.pivot_table(index='main_category',columns='currency',values='pledged',)
sns.heatmap(cco)


# In[ ]:


#ccd=dfx.pivot_table(index='pledged',columns='currency',values='pledged',)
#sns.heatmap(ccd)


# In[ ]:


dfx.pivot_table(index='main_category',columns='category',values='pledged')


# In[ ]:


cmp=dfx.pivot_table(index='category',columns='main_category',values='pledged')


# In[ ]:


cm=dfx.pivot_table(index='category',columns='main_category',values='state')


# In[ ]:


sns.heatmap(cm,cmap='coolwarm')


# In[ ]:


sns.heatmap(cmp,cmap='coolwarm',linecolor='black')


# In[ ]:


#dfx.head()


# In[ ]:


#dfx.head()


# In[ ]:


sns.countplot(x='country',data=dfx,hue='state',)


# In[ ]:


dfx['state'].count()


# In[ ]:


sns.countplot(x='currency',data=dfx,hue='state',)


# In[ ]:


sns.countplot(x='main_category',data=dfx,hue='state',)


# In[ ]:




