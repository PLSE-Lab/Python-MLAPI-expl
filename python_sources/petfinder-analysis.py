#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))


# In[ ]:


breed_labels = pd.read_csv("../input/breed_labels.csv")
color_labels = pd.read_csv("../input/color_labels.csv")
state_labels = pd.read_csv("../input/state_labels.csv")


# In[ ]:


test = pd.read_csv("../input/test/test.csv")
train = pd.read_csv("../input/train/train.csv")


# In[ ]:


breed_labels.info()


# In[ ]:


color_labels.info()


# In[ ]:


state_labels.info()


# In[ ]:


test.info()


# In[ ]:


train.info()


# In[ ]:


breed_labels.head()


# In[ ]:


color_labels.head()


# In[ ]:


state_labels.head()


# In[ ]:


test.head()


# In[ ]:


train.head()


# In[ ]:


len(test["Breed1"].unique())


# In[ ]:


test.corr()


# In[ ]:


f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(test.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


list1 = test['Gender']
list2 = test['Vaccinated']
z = zip(list1,list2)
print(z)
z_list = list(z)
print(z_list)


# In[ ]:


test.plot(kind='scatter', x='Color1', y='Color2',alpha = 1,color = 'red')
plt.xlabel('Color1')    
plt.ylabel('Color2')
plt.title('Color1 - Color2') 


# In[ ]:


test.Breed1.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()


# In[ ]:


test[(test['Breed1']<300) & (test['Fee']>100)]


# In[ ]:


test[["Breed1","Fee"]]


# In[ ]:


sns.lmplot(x='Breed1', y='Breed2', data=test)


# In[ ]:


plt.figure(figsize=(10,5))
sns.swarmplot(x='Type', y='Age', data=test)


# In[ ]:


plt.figure(figsize=(10,5))
sns.violinplot(x='Type',y='Age', data=test, inner=None)
sns.swarmplot(x='Type', y='Age', data=test, color='k', alpha=0.7) 
plt.title('Age by Type')


# In[ ]:


sns.pairplot(test, hue = 'Breed1')


# In[ ]:


labels = 'breed_labels', 'color_labels', 'state_labels', 'test'
sizes = [215, 130, 245, 210]
colors = ['gold', 'lightskyblue', 'red', 'lightcoral']
explode = (0.1, 0, 0, 0) 

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()


# In[ ]:




