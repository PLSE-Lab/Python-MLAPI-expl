#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
#plt.style.use('fivethirtyeight')
#import warnings
#warnings.filterwarnings('ignore')
import seaborn as sns


# In[ ]:


df=pd.DataFrame(pd.read_csv('../input/nba4.csv'))
df.info()


# In[ ]:


df['Salary'].describe(include='all')


# In[ ]:


print('',df['Weight'].min())
print('',df['Weight'].max())


# In[ ]:


print('',df['Height'].min())
print('',df['Height'].max())


# In[ ]:


df.columns =['Name','Team','Number','Position','Height','Weight','Age','Salary','College']
df.head(5)


# In[ ]:


df.notnull().sum()


# In[ ]:


df.isnull().sum()


# In[ ]:


df['Height']=(df['Height'].astype(str).str.replace('\'','.')).astype(float)
height=np.array(df['Height']).tolist()
data1=(df['Height'])/0.032
df['Height']=np.array(data1).astype(float)
pd.DataFrame.update
df.head(10)


# In[ ]:


np.std(df['Height'])


# In[ ]:


"mean of Weight",np.mean(df['Weight'])


# In[ ]:


df.fillna('NA',inplace=True)
df.head(10)


# In[ ]:


bins=[150,170,190,210,220,230,240,250,260,270,280,290]
df['wbinning']=pd.cut(df['Weight'],bins)
f,ax=plt.subplots(1,2,figsize=(18,8))
df['wbinning'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Weight')
ax[0].set_ylabel('')
sns.countplot('wbinning',data=df,ax=ax[1])
ax[1].set_title('Weight')
plt.xticks(rotation=90)
plt.show()


# In[ ]:



f,ax=plt.subplots(1,2,figsize=(18,8))
df['Salary'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Salary')
ax[0].set_ylabel('')
sns.countplot('Salary',data=df,ax=ax[1])
ax[1].set_title('Salary')
plt.show()


# In[ ]:


bins=[185,190,195,200,205,210,215,220,225,230,240,250,260,270]
df['hbinning']=pd.cut(df['Height'],bins)
f,ax=plt.subplots(1,2,figsize=(18,8))
df['hbinning'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Height')
ax[0].set_ylabel('')
sns.countplot('hbinning',data=df,ax=ax[1])
ax[1].set_title('Heights')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


pd.pivot_table(df, index=['Team','College','Name','Salary','Position'],aggfunc={'Height': lambda s: np.unique(s),'Weight': lambda s: np.unique(s)})


# In[ ]:


df = sns.load_dataset("Height","Weight")
x, y, hue = "Height", "Weight", "sex"
hue_order = ["Height", "Weight"]
f, axes = plt.subplots(1, 2)
sns.countplot(x=x, hue=hue, data=df, ax=axes[0])

