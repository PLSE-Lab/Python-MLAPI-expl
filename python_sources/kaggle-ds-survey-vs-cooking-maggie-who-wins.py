#!/usr/bin/env python
# coding: utf-8

# # Kaggle DS Survey vs Cooking Maggie ? Who Wins ?
# 
# A short story about the adventures of every Indian Engineering student.
# 
# ![2 min maggie](https://pics.me.me/2-minute-maggi-takes-8-10-minutes-to-cook-this-is-the-11683054.png)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Let's load the data set and see some statistics about people who have completed the survey within 5 minutes.
# 
# <img src='https://cdn1.vectorstock.com/i/1000x1000/05/50/timer-sign-icon-5-minutes-stopwatch-symbol-vector-3750550.jpg' height="350" width="250">

# In[ ]:


input_file_multi = '../input/multipleChoiceResponses.csv'


# In[ ]:


input_df = pd.read_csv(input_file_multi)

input_df.head()


# In[ ]:


input_df.columns


# In[ ]:


#duration_col = 'Time from Start to Finish (seconds)'


# ## Woah! First row is not actually the responce data. Let's remove that from our input dataset.

# In[ ]:


input_data = input_df.iloc[1:]
input_data.head()


# ## Why can't the column names be small and simple ?
# 
# ![Why, Data, Why](https://imgflip.com/s/meme/Jackie-Chan-WTF.jpg)

# In[ ]:


input_data = input_data.rename({"Time from Start to Finish (seconds)":"Duration"}, axis=1)
input_data.head()
# input_data.columns.values[0] = "Duration"
# input_data.head()


# In[ ]:


input_data.columns


# In[ ]:


# input_data.Duration


# In[ ]:


# input_data.iloc[:, 0]


# In[ ]:


input_data.head()


# In[ ]:


input_data['Duration'] = pd.to_numeric(input_data['Duration'])


# In[ ]:


input_data.head()


# ## Now that the duration column data is converted to numeric type, let's apply select and filter!

# In[ ]:


duration = 5 * 60
input_5min = input_data.loc[input_data['Duration'] <= duration]

input_5min.head()


# # Filtered the data. Now What ?

# In[ ]:


input_5min.shape


# In[ ]:


(3747/23859)*100


# 15.7% (3747 out of 23,859) participants completed survey within 5 minutes

# In[ ]:


input_5min.groupby('Q1')['Q1'].count()


# In[ ]:


#input_5min.groupby(['Q3','Q1'])['Q1'].count().sort_values(ascending=False)


# In[ ]:


input_5min.groupby(['Q3','Q1'])['Q1'].count().reset_index(name='count')                              .sort_values(['count'], ascending=False)


# In[ ]:


((625+555)/3747)*100


# * Men in India and US constitute upto 32% of the participants who have completed the survey within 5 minutes.
# 
# ## Let's work on finding out what they do ?

# ### Let's start with understanding Indian audience :)

# In[ ]:


input_india = input_5min.loc[input_5min['Q3'] == 'India']
input_india.head()


# In[ ]:


input_india.shape


# In[ ]:


input_india.groupby(['Q7', 'Q1'])['Q1'].count().reset_index(name='count')                               .sort_values(['count'], ascending=False)


# In[ ]:


print(230/760 * 100)
print(106/760 * 100)


# About 30% of Indian respondants are Male students.
# 
# 14% of the male respondants are in IT field.

# ## Let's see what course are the male students pursuing.

# In[ ]:


input_in_male_stud = input_india.loc[(input_india['Q7'] == 'I am a student') &
                                    (input_india['Q1'] == 'Male')]

input_in_male_stud.head()


# In[ ]:


input_in_male_stud.shape


# In[ ]:


input_in_male_stud.groupby(['Q4', 'Q5'])['Q5'].count().reset_index(name='count')                            .sort_values(['count'], ascending=False)


# In[ ]:


print(95/230 * 100)
print((95+30)/230 * 100)


# 41% of the Male students from India are pursuing Bachelor's degree in C.S.
# 
# More than half of the respondants are pursuing C.S. degree (Bachelor's and Master's included)

# In[ ]:


input_india.shape


# In[ ]:


input_india.groupby(['Q4', 'Q5'])['Q5'].count().reset_index(name='count')                            .sort_values(['count'], ascending=False)


# In[ ]:


print((243+121)/762 * 100)


# Almost half of the Indian Students, pursuing Computer Science degree, completed taking the Kaggle Survey quicker than the time it takes to prepare Maggi in hostels ;)

# In[ ]:




