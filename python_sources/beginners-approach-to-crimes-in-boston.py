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


# # Boston Crime & Data Analysis

# Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium


# ### Reading Data

# In[ ]:


Crimes=pd.read_csv('../input/crimes-in-boston/crime.csv',encoding='latin-1')
Crimes.head()


# ### Understanding the Data 

# In[ ]:


Crimes.info()


# In[ ]:


Crimes.describe()


# ### Data Cleaning

# In[ ]:


Crimes.columns


# In[ ]:


Crimes.isnull().sum()


# #### Removing Null Values

# In[ ]:


Crimes.drop(columns='SHOOTING',inplace=True)


# ### No Of Crimes By Years 

# In[ ]:


sns.countplot(data=Crimes,x='YEAR')
plt.show()


# ### Trends in crimes by different Months

# In[ ]:


sns.countplot(data=Crimes,x='MONTH')
plt.show()


# ### No of crimes by Different Days.

# In[ ]:


sns.countplot(data=Crimes, x='DAY_OF_WEEK')
plt.show()


# In[ ]:


sns.catplot(data=Crimes, y='OFFENSE_CODE_GROUP',height=10,kind='count',
            aspect=1.5)
plt.show()


# In[ ]:


from wordcloud import WordCloud

text = []
for i in Crimes.OFFENSE_CODE_GROUP:
    text.append(i)#here we are adding word to text array but it's looking like this ['Larency','Homicide','Robbery']
text = ''.join(map(str, text)) #Now we make all of them like this [LarencyHomicideRobbery]

wordcloud = WordCloud(width=1600, height=800, max_font_size=300).generate(text)
plt.figure(figsize=(20,17))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:




