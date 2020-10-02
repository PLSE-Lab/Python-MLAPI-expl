#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv("/kaggle/input/golden-globe-awards/golden_globe_awards.csv")
df.head(10)


# **NO of null val**

# In[ ]:


df.isnull().sum()


# **Dimension of data set**

# In[ ]:


df.shape


# **Total unique category of awards**

# In[ ]:


df['category'].nunique()


# **Number of unique films in the dataset**

# In[ ]:


print('Number of unique films in the award ceremony: ', df['film'].nunique())


# **Nominee who was nominated the most number of times**

# In[ ]:


most_nominated_nominee = df.groupby('nominee')['win'].count()
pd.DataFrame(most_nominated_nominee[most_nominated_nominee == most_nominated_nominee.max()])


# **Nominees who won atlest 5 awards**

# In[ ]:


more_5_win = df.groupby('nominee')['win'].sum()
more_5_win = more_5_win.reset_index()
more_5_win = more_5_win[more_5_win['win'] >= 5].sort_values(ascending=False, by='win')

# print('There are ', more_5_win['win'].count(), 'nominees who won more than 5 awards')
plt.figure(figsize=(20,8))
sns.set_style('whitegrid')
sns.barplot(x='nominee', y='win', data=more_5_win, palette='hls')
plt.title('Nominees who won more than 5 awards', fontsize=14)
plt.xlabel('Nominee', fontsize=14)
plt.ylabel('Total Wins', fontsize=14)
plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12)
plt.show()


# **Nominees who were nominated more than 15 times**

# In[ ]:


more_15_nominated = df.groupby('nominee')['win'].count()
more_15_nominated = more_15_nominated.reset_index()
more_15_nominated = more_15_nominated[more_15_nominated['win'] >= 15].sort_values(ascending=False, by='win')

# top_10_nominated
plt.figure(figsize=(20,8))
sns.set_style('whitegrid')
sns.barplot(x='nominee', y='win', data=more_15_nominated, palette='hls')
plt.title('Nominees who were nominated more than 15 times', fontsize=14)
plt.xlabel('Nominee', fontsize=14)
plt.ylabel('Nomination Count', fontsize=14)
plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12)
plt.show()


# **Film won most award :-**

# In[ ]:


film_most_awards = df.groupby('film')['win'].sum()
pd.DataFrame(film_most_awards[film_most_awards == film_most_awards.max()])


# **Films/T.V. Show that have won atleast 5 awards**

# In[ ]:


more_5_films = df.groupby('film')['win'].sum()
more_5_films = more_5_films.reset_index()
more_5_films = more_5_films[more_5_films['win'] >= 5].sort_values(ascending=False, by='win')

plt.figure(figsize=(20,8))
sns.set_style('whitegrid')
sns.barplot(x='film', y='win', data=more_5_films, palette='hls')
plt.title('Films that have won atleast 5 awards', fontsize=12)
plt.xlabel('Film', fontsize=12)

plt.ylabel('Award Count', fontsize=12)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(rotation=90, fontsize=12)
plt.show()


# **Film that has won most awards in a single year**

# In[ ]:


film_awards_year = df.groupby(['film', 'year_award'])['win'].sum()
pd.DataFrame(film_awards_year[film_awards_year == film_awards_year.max()])


# **Film that got the most nominations in a single year**

# In[ ]:


film_nominations = df.groupby(['film', 'year_award'])['win'].count()
pd.DataFrame(film_nominations[film_nominations == film_nominations.max()]).reset_index()


# ****Most nominated directors in the Motion Picture category****

# In[ ]:


director_motion_picture = df[df['category'].str.contains('Best Director - Motion Picture')]
director_motion_picture = director_motion_picture.groupby('nominee')['win'].count().reset_index().sort_values(ascending=False, by='win')
director_motion_picture = director_motion_picture[director_motion_picture['win'] >= 5]
# director_motion_picture

plt.figure(figsize=(15,6))
sns.set_style('whitegrid')
sns.barplot(x='nominee', y='win', data=director_motion_picture, palette='hls')
plt.title('Most nominated directors in motion picture category', fontsize=12)
plt.xlabel('Director', fontsize=12)

plt.ylabel('Nominations', fontsize=12)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(rotation=90, fontsize=12)
plt.show()


# ** NOMINATIONS AND AWARDS HAVE CHANGED OVER THE YEARS**

# In[ ]:


df_year = pd.DataFrame()
df_year['total_nominations'] = df.groupby(['year_award'])['win'].count()
df_year['wins'] = df.groupby(['year_award'])['win'].sum()
plt.plot(df_year.index, df_year['total_nominations'], label = 'nominations')
plt.plot(df_year.index, df_year['wins'], label = 'wins')
plt.legend()


# In[ ]:




