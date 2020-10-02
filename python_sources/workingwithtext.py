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


# # Working with text in pandas

# In[ ]:


sentence = ['Monday: Going to college at 10:30 am',
           'Tuesday: Going to job at 7:20 am',
           'Wednesday: At 8:20 am there is a cricket game',
           'Thursday: Be at home for 5:30 pm',
           'Friday: Going to school at 8:20 am and returning at 5:15 pm']
df = pd.DataFrame(sentence,columns=['Routine'])
df


# Finding the number of character for each string in dataframe text.

# In[ ]:


df['Routine'].str.len()


# Finding the number of tokens for each string in data frame.

# In[ ]:


df['Routine'].str.split().str.len()


# Find which items contains word Going

# In[ ]:


df[df['Routine'].str.contains('Going')]


# Find how many times the digits occures in the string

# In[ ]:


df['Routine'].str.findall('\d')


# Group and find the hours and minutes

# In[ ]:


df['Routine'].str.findall(r'(\d?\d):(\d\d)')


# Replace week days with ***

# In[ ]:


df['Routine'].str.replace(r'\w+day\b', '***')


# Replace week days with three letter abbribation

# In[ ]:


df['Routine'].str.replace(r'(\w+day\b)', lambda x: x.groups()[0][:3])


# New column for first match of extracted columns

# In[ ]:


df['Routine'].str.extract(r'(\d?\d):(\d\d)')


# Extract the entire time, the hours, the minutes, and the period

# In[ ]:


df['Routine'].str.extract(r'((\d?\d):(\d\d) ?([ap]m))')


# Extract the entire time, the hours, the minutes, and the period with group names

# In[ ]:


df['Routine'].str.extractall(r'(?P<time>(?P<hour>\d?\d):(?P<minute>\d\d) ?(?P<period>[ap]m))')


# In[ ]:




