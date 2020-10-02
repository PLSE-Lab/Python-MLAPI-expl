#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


my_data = pd.read_csv('../input/young-people-survey/responses.csv')
my_data.head()


# # correlation coffecint between each question in the field and whole field.

# In[ ]:


music_data = my_data.iloc[:,0:19]
music_data.columns


# In[ ]:


print(music_data.isnull().sum())


# In[ ]:


music_data.dropna(inplace = True)
music_data.reset_index(drop=True,inplace=True)
row = len(music_data.index)


# In[ ]:


music_data.info()


# In[ ]:


corr1 = music_data.corr()
print (corr1)


# In[ ]:


plt.figure(figsize= (16,6))
sns.heatmap(corr1)


# In[ ]:


movie_data = my_data.iloc[:,19:31]
movie_data.columns


# In[ ]:


print(movie_data.isnull().sum())


# In[ ]:


movie_data.dropna(inplace = True)
movie_data.reset_index(drop=True,inplace=True)
row = len(movie_data.index)
movie_data.info()


# In[ ]:


corr2 = movie_data.corr()
print (corr2)


# In[ ]:


plt.figure(figsize= (16,6))
sns.heatmap(corr2)


# In[ ]:


hobbies_interests_data = my_data.iloc[:,31:63]
hobbies_interests_data.columns


# In[ ]:


print(hobbies_interests_data.isnull().sum())


# In[ ]:


hobbies_interests_data.dropna(inplace = True)
hobbies_interests_data.reset_index(drop=True,inplace=True)
row = len(hobbies_interests_data.index)
hobbies_interests_data.info()


# In[ ]:


corr3 = hobbies_interests_data.corr()
print (corr3)


# In[ ]:


plt.figure(figsize= (16,6))
sns.heatmap(corr3)


# In[ ]:


phobias_data = my_data.iloc[:,63:73]
phobias_data.columns


# In[ ]:


print(phobias_data.isnull().sum())


# In[ ]:


phobias_data.dropna(inplace = True)
phobias_data.reset_index(drop=True,inplace=True)
row = len(phobias_data.index)
phobias_data.info()


# In[ ]:


corr4 = phobias_data.corr()
print (corr4)


# In[ ]:


plt.figure(figsize= (16,6))
sns.heatmap(corr4)


# In[ ]:


health_habits_data = my_data.iloc[:,73:76]
health_habits_data.columns


# In[ ]:


print(health_habits_data.isnull().sum())


# In[ ]:


health_habits_data.dropna(inplace = True)
health_habits_data.reset_index(drop=True,inplace=True)
row = len(health_habits_data.index)
health_habits_data.info()


# In[ ]:


s = (health_habits_data.dtypes == 'object')
object_cols = list(s[s].index)
from sklearn.preprocessing import LabelEncoder
label_health_habits_data = health_habits_data.copy()
label_encoder = LabelEncoder()
for col in object_cols:
    label_health_habits_data[col] = label_encoder.fit_transform(health_habits_data[col])


# In[ ]:


corr5 = label_health_habits_data.corr()
print (corr5)


# In[ ]:


plt.figure(figsize= (6,4))
sns.heatmap(corr5)


# In[ ]:


personality_data = my_data.iloc[:,76:133]
personality_data.columns


# In[ ]:


print(personality_data.isnull().sum())


# In[ ]:


personality_data.dropna(inplace = True)
personality_data.reset_index(drop=True,inplace=True)
row = len(personality_data.index)
personality_data.info()


# In[ ]:


s = (personality_data.dtypes == 'object')
object_cols = list(s[s].index)
from sklearn.preprocessing import LabelEncoder
label_personality_data = personality_data.copy()
label_encoder = LabelEncoder()
for col in object_cols:
    label_personality_data[col] = label_encoder.fit_transform(personality_data[col])


# In[ ]:


corr6 = label_personality_data.corr()
print (corr6)


# In[ ]:


plt.figure(figsize= (20,10))
sns.heatmap(corr6)


# In[ ]:


spending_habits_data = my_data.iloc[:,133:140]
spending_habits_data.columns


# In[ ]:


print(spending_habits_data.isnull().sum())


# In[ ]:


spending_habits_data.dropna(inplace = True)
spending_habits_data.reset_index(drop=True,inplace=True)
row = len(spending_habits_data.index)
spending_habits_data.info()


# In[ ]:


corr7 = spending_habits_data.corr()
print (corr7)


# In[ ]:


sns.heatmap(corr7)


# In[ ]:


demographics_data = my_data.iloc[:,140:150]
demographics_data.columns


# In[ ]:


print(demographics_data.isnull().sum())


# In[ ]:


demographics_data.dropna(inplace = True)
demographics_data.reset_index(drop=True,inplace=True)
row = len(demographics_data.index)
demographics_data.info()


# In[ ]:


s = (demographics_data.dtypes == 'object')
object_cols = list(s[s].index)
from sklearn.preprocessing import LabelEncoder
label_demographics_data = demographics_data.copy()
label_encoder = LabelEncoder()
for col in object_cols:
    label_demographics_data[col] = label_encoder.fit_transform(demographics_data[col])


# In[ ]:


corr8 = label_demographics_data.corr()
print (corr8)


# In[ ]:


sns.heatmap(corr8)


# In[ ]:


my_data.dropna(inplace = True)
my_data.reset_index(drop=True,inplace=True)
row = len(my_data.index)
my_data.info()
   


# In[ ]:


s = (my_data.dtypes == 'object')
object_cols = list(s[s].index)
from sklearn.preprocessing import LabelEncoder
label_my_data = my_data.copy()
label_encoder = LabelEncoder()
for col in object_cols:
    label_my_data[col] = label_encoder.fit_transform(my_data[col])


# In[ ]:




