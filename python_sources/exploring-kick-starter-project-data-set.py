#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# For now i am just posting the visulaizations of the data set, soon will be posting some models and feature extractions.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

d = pd.read_csv('../input/original-data-set/dspro.csv',encoding='latin1')


df = pd.DataFrame(d)


# In[ ]:


df.head()


# In[ ]:


#mapping the state of kick starter project to 5 classes
df['state']=df['state'].map({'failed':0,'successful':1,'canceled':2,'live':3,'undefined':4,'suspended':5})


# In[ ]:


df.head()


# In[ ]:





# In[ ]:





# In[ ]:


sns.jointplot(x=df['state'],y=df['backers'],data=df)


# In[ ]:


#we are considering only Sucessful & Failed Projects, we are only visualizing sucessful & failed because
# other classes are very low in number so i am assuming it will be not enough to train the data model 
dfx=df[df['state']<2]
df.head()
sns.barplot(x='state',y='backers',data=dfx,hue='state')


# In[ ]:


sns.barplot(x='state',y='backers',data=df,hue='state',estimator=np.std)


# In[ ]:


sns.barplot(x='state',y='pledged',data=df,hue='state')


# In[ ]:


sns.barplot(x='state',y='goal',data=dfx,hue='state')


# In[ ]:


sns.barplot(x='state',y='goal',data=df,hue='state')


# In[ ]:


# We can see here count of class 2,3,4,5 which are cancelled, live, undefined & suspended 
# Count of them is limited when compared to class 0 & 1 ie sucess & failure classes.
sns.countplot(x='state',data=df)


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


dfx.head()


# In[ ]:


#dfx.head()


# In[ ]:


sns.countplot(x='main_category',data=dfx,hue='state',)


# In[ ]:


dfx['state'].count()


# In[ ]:


# Data is mostly related to USA so we can consider cleaning or droping other countries
sns.countplot(x='country',data=dfx,hue='state',)


# In[ ]:


# Most of the transactions are in US Dollars. 
sns.countplot(x='currency',data=dfx,hue='state',)


# In[ ]:




