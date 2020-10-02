#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import matplotlib.pyplot as plt # plotting
import seaborn as sn # plotting

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # ABOUT THE DATA
# I generated this data randomly, using Excel's INDEX, RANDBETWEEN and COUNTA functions. For instance, to generate the gender columns, I used the command =INDEX(J$3:J$4,RANDBETWEEN(1,COUNTA(J$3:J$4)),1), where J3 and J4 contain male and female. A similar formula extension was applied across all columns. 

# In[ ]:


df=pd.read_excel('/kaggle/input/web-analytics/WebsiteData.xlsx')
df


# # ONE DIMENSIONAL EDA

# In[ ]:


fig, ax = plt.subplots()
df['Gender'].value_counts().plot(ax=ax, kind='bar')


# In[ ]:


fig, ax = plt.subplots()
df['Location'].value_counts().plot(ax=ax, kind='bar')


# In[ ]:


fig, ax = plt.subplots()
df['Ethnicity'].value_counts().plot(ax=ax, kind='bar')


# In[ ]:


fig, ax = plt.subplots()
df['Language'].value_counts().plot(ax=ax, kind='bar')


# In[ ]:


fig, ax = plt.subplots()
df['Affinity Category'].value_counts().plot(ax=ax, kind='bar')


# In[ ]:


fig, ax = plt.subplots()
df['New vs Exisitng'].value_counts().plot(ax=ax, kind='bar')


# In[ ]:


fig, ax = plt.subplots()
df['Traffic Source'].value_counts().plot(ax=ax, kind='bar')


# In[ ]:


df[['Age']].plot(kind='hist',bins=[0,20,40,60,80,100],rwidth=0.8)
plt.show()


# In[ ]:


df[['Engagement']].plot(kind='hist',bins=[0,20,40,60,80,100],rwidth=0.8)
plt.show()


# # TWO DIMENSIONAL EDA

# In[ ]:


#One hot encoding of categorical data
dfe = pd.get_dummies(df)
dfe


# In[ ]:


#Establishing correlation
corrMatrix = dfe.corr()
plt.figure(figsize = (40,40))
ax=sn.heatmap(corrMatrix, annot=True)
plt.show()


# In[ ]:


upper = corrMatrix.where(np.triu(np.ones(corrMatrix.shape), k=1).astype(np.bool))
s = upper.unstack()


# In[ ]:


#Establishing strong positive correlations
so = s.sort_values(kind="quicksort")
sodf=pd.Series.to_frame(so)
psodf=sodf.loc[0.0285<sodf[0]]
psodf=psodf.loc[sodf[0]<1]
psodf


# In[ ]:


#Establishing strong negative correlations
nsodf=sodf.loc[-0.027>sodf[0]]
nsodf=nsodf.loc[sodf[0]>-0.1]
nsodf


# In[ ]:


sn.regplot(data=df.sample(n=1000),x ='Age', y='Engagement')


# In[ ]:


sn.swarmplot(x="Gender", y="Engagement", data=df)


# In[ ]:


sn.swarmplot(x="Location", y="Engagement", data=df)


# In[ ]:


sn.swarmplot(x="Ethnicity", y="Engagement", data=df)


# In[ ]:


sn.swarmplot(x="Language", y="Engagement", data=df)


# In[ ]:


sn.swarmplot(x="Affinity Category", y="Engagement", data=df)


# In[ ]:


sn.swarmplot(x="New vs Exisitng", y="Engagement", data=df)


# In[ ]:


sn.swarmplot(x="Traffic Source", y="Engagement", data=df)

