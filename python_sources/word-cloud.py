#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import sqlite3

db = sqlite3.connect('../input/database.sqlite')


# In[ ]:


df = pd.read_sql_query('SELECT Name, Gender, Count from NationalNames WHERE Year = 2014 AND Name != "" AND Count != ""', db)
print(df.head())


# In[ ]:


df.Name.value_counts(sort=True, dropna=False)


# In[ ]:


# Some of the names occur twice in the dataset because counts are recorded for both genders
df[df.Name.isin(['Amore', 'Ellis', 'Trinidad', 'Nikita', 'Berlin'])]


# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
plt.figure(figsize=(15,15))
plt.axis('off')
    
def name_cloud(df):
    # WordCloud takes a string, creating a name string for all occurences of name
    name_string = ''
    for row in df.itertuples():
        for x in range(0, row.Count):
            name_string += " " + str(row.Name)
        
        # Clean string to leave only letters
        name_string = ''.join([char for char in name_string if (char.isalpha() or char == ' ')])
       
    # Create wordcloud
    wordcloud = WordCloud(background_color='white')
    wordcloud.generate(name_string)
    
    return wordcloud


# In[ ]:


# Gender independent name word cloud
ind_cloud = name_cloud(df)
plt.imshow(wordcloud)
plt.show()


# In[ ]:


# Males name word cloud
name_cloud(df[df.Gender = 'M'])


# In[ ]:


# Females name word cloud
name_cloud(df[df.Gender = 'F'])


# In[ ]:




