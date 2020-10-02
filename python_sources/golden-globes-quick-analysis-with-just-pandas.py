#!/usr/bin/env python
# coding: utf-8

# This is a beginner friendly notebook. We spend so much time performing analysis, making complicated models and tuning parameters for neural networks. But often times, a lot of the questions we want to answer can be tackle with just simple queries in SQL / Pandas without using such complicated models. In this notebook, we use only pandas to do quick analysis and address many 1st level questions to get a big picture about golden globe awards.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("/kaggle/input/golden-globe-awards/golden_globe_awards.csv")


# ## Look at data / General Description

# In[ ]:


df.head()


# In[ ]:


# Data Types for each column

df.dtypes


# In[ ]:


# "Film" feature has some missing values
df.isnull().any()


# In[ ]:


# Fill missing film names with "Unknown"
df.film.fillna('Unknown', inplace=True)


# In[ ]:


df.isnull().any()


# ## How many awards were given out each year?

# In[ ]:


win_num_by_year = df[df.win==True].groupby('year_award').win.count().to_frame()
win_num_by_year


# It used to be less than 10 awards given in the first four years. Then, it increased a lot over the years. 

# In[ ]:


win_num_by_year.query('win >= 12 & win < 25')


# In[ ]:


win_num_by_year.query('win == 25')


# Starting from the 1990s, 24 awards were more consistently given out and from 2007, 25 awards were given consistently.

# ## Who are the top 3 actors/actresses who won the most golden globes?

# In[ ]:


df[df.win==True].groupby('nominee').count().sort_values('win', ascending=False).head(3)


# They are Meryl Streep, Jane Fonda and Barbra Sreisand!

# ## Which categories have the highest probability of winning golden globes once you get nominated?

# In[ ]:


df.groupby('category').win.apply(lambda x: sum(x==True)*100/x.count()).to_frame().sort_values('win',ascending=False).head(20)


# There are some categories where you win the award for sure once you get nominated (e.g. Hollywood citizens award, New Foreign Star Of The Year - Actor etc.). Categories such as Actor / Actress In A Leading Role, Picture and Cinematography have pretty high probability of winning once you get nominated (> 70%).

# ## Which film earned the most awards?

# In[ ]:


df[df.win==True].groupby('film').win.count().to_frame().sort_values('win').tail(10)


# This might be misleading because there were a lot of missing values for "film names" and I just filled them all as "unknown". Nevertheless, based on the data we have, MASH, Alice, Carol Burnett Show were received the most Golden Globes awards followed by La La Land and Lawrence of Arabia.

# ## Who won the Supporting Role in any Motion Picture awards the most?

# Often times, the importance of supporting roles is overlooked.

# In[ ]:


df[df.category=='Best Performance by an Actress in a Supporting Role in any Motion Picture'].groupby('nominee').win.count().to_frame().sort_values('win').tail(10)


# In[ ]:


df[df.category=='Best Performance by an Actor in a Supporting Role in any Motion Picture'].groupby('nominee').win.count().to_frame().sort_values('win').tail(10)


# ### Is there any correlation between length of title of film and its probability of winning awards?

# Just out of curiosity (but expecting the correlation to be very weak)

# In[ ]:


# Getting word count of film title
df['film_word_count'] = df.film.str.split(" ").apply(lambda x: len(x))

# Replace True or False to 1 or 0
df.win.replace({True: 1, False: 0}, inplace=True)


# In[ ]:


df[['win','film_word_count']].corr().iloc[0,1]


# Correlation, jsut as we expected, is very weak
