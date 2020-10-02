#!/usr/bin/env python
# coding: utf-8

# # Load data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv')


# In[ ]:


list(df.columns)


# In[ ]:


df.head()


#  
#   
#    
#    
#    
#    
#    
#    

#  

# # Exploratory data analysis

# In[ ]:


len(df.columns)


# In[ ]:


df.info()


# ### Numerical and categorical variables

# In[ ]:


type(df.info())


# In[ ]:


df['GameWeather'].unique()


# In[ ]:


len(df['GameWeather'].unique())


# In[ ]:


num_variable = [
    'GameId', # 512 games
    'PlayId', # 23171 play
    'X',
    'Y',
    'S',
    'A',
    'Dis',
    'Orientation',
    'Dir',
    'NflId', # 2231 values
    # 'JerseyNumber', # 99 numbers
    
]

cat_variables = [
    'Team', # 2 values: Home or away
    'DisplayName', # 2230
    'Season', # two values: 2017, 2018
    'YardLine', # 50 values
    ''
]


# # Missing values

# In[ ]:


import missingno as msno


# Number of missing values:

# In[ ]:


msno.bar(df);


# Pattern of missing values:

# In[ ]:


msno.matrix(df);


# In[ ]:


msno.matrix(df.sort_values('GameWeather'))


# As we can see the temperature missing values is highly correlated with and wind speed and direction:

# In[ ]:


msno.heatmap(df)


# In[ ]:


msno.dendrogram(df);


# In[ ]:




