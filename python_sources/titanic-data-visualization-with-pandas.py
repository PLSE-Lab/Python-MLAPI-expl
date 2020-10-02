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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Import Required Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('classic')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Import Dataset

# In[ ]:


df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.head()


# # Histogram/Distribution plot

# In[ ]:


ax = df['Age'].hist(bins=25)
ax.set_xlabel('Age')
ax.set_ylabel('Count')
ax.set_title('Distribution by Age')


# # Histogram with categorical variables

# In[ ]:


ax = df['Age'][df['Sex']=='male'].hist(bins=25, label='Male')
df['Age'][df['Sex']=='female'].hist(bins=25, ax = ax, label='Female')
ax.set_xlabel('Age')
ax.set_ylabel('Count')
ax.set_title('Distribution by Age')
ax.legend()


# # KDE (Kernel Density Estimation) plot

# In[ ]:



ax = df['Age'][df['Sex']=='male'].plot.kde( label='Male')
df['Age'][df['Sex']=='female'].plot.kde(ax = ax, label='Female')
ax.set_xlabel('Age')
ax.set_ylabel('Density')
ax.set_title('Distribution by Age')
ax.legend()


# # Vertical bar plot

# In[ ]:


pclass_count =df['Pclass'].value_counts()
plt.figure(figsize=(4,5))
ax = pclass_count.plot.bar()
ax.set_ylabel('Count', rotation=90)
ax.set_xlabel('Passenger Class')
ax.set_title('Pclass vs Count')


# # Horizontal bar plot

# In[ ]:


pclass_count =df['Embarked'].value_counts()
plt.figure(figsize=(4,4))
ax = pclass_count.plot.barh()
ax.set_ylabel('Emabrked At', rotation=90)
ax.set_xlabel('Count')
ax.set_title('Embarked vs Count')


# # Stacked bar plot

# In[ ]:


# reference https://stackoverflow.com/questions/23415500/pandas-plotting-a-stacked-bar-chart

df.groupby(['Sex','Survived']).size().unstack().plot(kind='bar', stacked=True)
plt.ylabel('Count')
plt.title('Survival rate by Sex')


# # Boxplot

# In[ ]:


df.boxplot(by='Pclass', column=['Age'])


# # Stacked bar plot for missing values 

# In[ ]:


plt.figure(figsize=(10,10))
missing_count= df.isnull().apply(pd.value_counts).fillna(0).transpose()
missing_count.rename({False: 'Values present', True:'Values missing'}, axis=1,inplace=True)
ax = missing_count.plot.bar(stacked=True)
plt.title("Missing values by columns")
ax.legend(loc=10)
plt.xlabel("Column names")
plt.ylabel('Count')

