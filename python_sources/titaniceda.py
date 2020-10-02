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


train = pd.read_csv('../input/train.csv')
gender_submission = pd.read_csv('../input/gender_submission.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.info()


# In[ ]:


gender_submission.info()


# In[ ]:


test.info()


# In[ ]:


train.head()


# In[ ]:


gender_submission.head()


# In[ ]:


test.head()


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# In[ ]:


# Boxplot
plt.figure(figsize=(10,5))
sns.boxplot(data=train)


# In[ ]:


plt.figure(figsize=(10,5))
# Count Plot
sns.countplot(x='Pclass', data=train)
 
# Rotate x-labels
plt.xticks(rotation=0)


# In[ ]:


# Density Plot
sns.kdeplot(train.Pclass, train.Survived)


# In[ ]:


# Joint Distribution Plot
sns.jointplot(x='Pclass', y='Survived', data=train)


# In[ ]:


sns.pairplot(test, hue = 'Sex')


# In[ ]:


plt.figure(figsize=(10,5))
sns.lineplot(x="Pclass", y="Fare", hue="Survived", style="Sex", data=train)


# In[ ]:


plt.figure(figsize=(10,10))
g = sns.lmplot(x="Age", y="Pclass", hue="Sex",truncate=True, height=5, data=test)
g.set_axis_labels("Age", "Class")


# In[ ]:


labels = 'Pclass', 'Age', 'Sex', 'Survived'
sizes = [215, 130, 245, 210]
colors = ['gold', 'lightskyblue', 'red', 'lightcoral']
explode = (0.1, 0, 0, 0) 

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()


# In[ ]:


sns.set()

r = np.linspace(0, 10, num=100)
test = pd.DataFrame({'r': r, 'slow': r, 'medium': 2 * r, 'fast': 4 * r})

test = pd.melt(test, id_vars=['r'], var_name='Pclass', value_name='Age')

g = sns.FacetGrid(test, col="Pclass", hue="Pclass", subplot_kws=dict(projection='polar'), height=4.5, sharex=False, sharey=False, despine=False)

g.map(sns.scatterplot, "Age", "r")


# In[ ]:




