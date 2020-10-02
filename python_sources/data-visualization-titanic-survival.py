#!/usr/bin/env python
# coding: utf-8

# Libraries involved in this Kernel are 
# 
# **Pandas** for data manipulation and ingestion.
# 
# **Matplotlib** and **Seaborn** for data visualization
# 
# 

# ### Import libraries

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pylab as plot


# # Load the data

# In[ ]:


data = pd.read_csv("../input/train.csv")
print(data.shape)


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data['Age'] = data['Age'].fillna(data['Age'].median())


# **Visualizing survival based on the gender.**

# In[ ]:


data['Died'] = 1 - data['Survived']
data.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7),
                                                          stacked=True, colors=['g', 'r']);


# In[ ]:


data.groupby('Sex').agg('mean')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7), 
                                                           stacked=True, colors=['g', 'r']);


# **Inference : Women are more likely to survive**

# **Visualizing survival based on the age**

# In[ ]:


fig = plt.figure(figsize=(25, 7))
sns.violinplot(x='Sex', y='Age', 
               hue='Survived', data=data, 
               split=True,
               palette={0: "r", 1: "g"}
              );


# **Inference :**
# 
# **Younger male tend to survive**
# 
# **A large number of passengers between 20 and 40 die**
# 
# **The age doesn't seem to have a direct impact on the female survival**

# **Visualizing survival based on the fare ticket**

# In[ ]:


figure = plt.figure(figsize=(25, 7))
plt.hist([data[data['Survived'] == 1]['Fare'], data[data['Survived'] == 0]['Fare']], 
         stacked=True, color = ['g','r'],
         bins = 50, label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend();


# **Inference : Passengers with cheaper ticket fares are more likely to die.**

# **Visualizing age,survival and fare on a single chart**

# In[ ]:


plt.figure(figsize=(25, 7))
ax = plt.subplot()

ax.scatter(data[data['Survived'] == 1]['Age'], data[data['Survived'] == 1]['Fare'], 
           c='green', s=data[data['Survived'] == 1]['Fare'])
ax.scatter(data[data['Survived'] == 0]['Age'], data[data['Survived'] == 0]['Fare'], 
           c='red', s=data[data['Survived'] == 0]['Fare']);


# **Correlation between Ticket Fare and Classes**

# In[ ]:


ax = plt.subplot()
ax.set_ylabel('Average fare')
data.groupby('Pclass').mean()['Fare'].plot(kind='bar', figsize=(25, 7), ax = ax);


# **Visualizing survival based on the embarkation.**

# In[ ]:


fig = plt.figure(figsize=(25, 7))
sns.violinplot(x='Embarked', y='Fare', hue='Survived', data=data, split=True, palette={0: "r", 1: "g"});


# **Inference : The embarkation C have a wider range of fare tickets and therefore the passengers who pay the highest prices are those who survive.**

# ### References
# https://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html
