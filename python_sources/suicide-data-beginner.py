#!/usr/bin/env python
# coding: utf-8

# **Suicide** is the act intentionally causing one's own *death*. Close to **800 000 people** die due to suicide *every year*, which is one person *every 40 seconds*. **Suicide** is a *global phenomenon* and *occurs throughout the lifespan*. *Effective and evidence-based interventions* can be implemented at population, sub-population and individual levels to **prevent** *suicide and suicide attempts*. There are indications that for *each adult who died by suicide there may have been more than 20 others attempting suicide.*

# **Importing libraries **

# In[ ]:


import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
print(os.listdir("../input"))


# In[ ]:


dataset = pd.read_csv('../input/master.csv')


# In[ ]:


#first 5 rows
dataset.head()


# In[ ]:


#last 5 rows
dataset.tail()


# In[ ]:


#random 5 rows
dataset.sample(5)


# In[ ]:


unique_country = dataset['country'].unique()
print(unique_country)
#unique country


# **Let's check for country**

# In[ ]:


#Info about the dataset
dataset.info()


# In[ ]:


#dataset Column
dataset.columns


# In[ ]:


#dataset shape
print('Data shape')
dataset.shape


# In[ ]:


#null value check
dataset.isnull().any()


# In[ ]:


dataset.isnull().values.any()


# In[ ]:


dataset.isnull().sum()


# In[ ]:


plt.figure(figsize=(10,25))
sns.countplot(y='country', data=dataset, alpha=0.7)
plt.title('Date by country')
plt.show()


# **Lets see the gender**

# In[ ]:


plt.figure(figsize=(16,7))
#Plot the graph
sex = sns.countplot(x='sex', data=dataset)


# In[ ]:


plt.figure(figsize=(16,7))
cor = sns.heatmap(dataset.corr(), annot =True)


# In[ ]:


plt.figure(figsize=(16,7))
bar = sns.barplot(x='sex', y = 'suicides_no', hue='age', data= dataset)


# In[ ]:


plt.figure(figsize=(16,7))
bar = sns.barplot(x='sex', y = 'suicides_no', hue='generation', data= dataset)


# **Age group distribution**

# In[ ]:


dataset.groupby('age')['sex'].count()


# In[ ]:


sns.barplot(x=dataset.groupby('age')['sex'].count().index, y=dataset.groupby('age')['sex'].count().values)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(dataset.sex,hue=dataset.age)
plt.title('Sex and age')
plt.show()


# Thank you

# In[ ]:




