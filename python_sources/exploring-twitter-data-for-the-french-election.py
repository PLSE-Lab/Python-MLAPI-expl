#!/usr/bin/env python
# coding: utf-8

# Exploring Twitter data for the first round of the French Presidential Election

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


#start by importing the necessary libraries to play with the data

import pandas as pd
import numpy as np
import sqlite3

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#first we need to load the sqlite files into a pandas dataframe
#remember to open and close the sqlite connection, and that we have multiple databases

#first create a list to help with importing the files

numlist = [str(i)+"_"+str(j) for i in range(11,18) for j in range(0,2)]

#remove ones that aren't there, add google grouping

numlist.remove('11_0')
numlist.remove('17_1')
#numlist.append('googletrends') deal with google trends later

#create a list to house the giant collection of dfs

df_list = []

for x in numlist:
    base = "../input/database_"
    path = base + x + ".sqlite"
    connection = sqlite3.connect(path)
    df_list.append(pd.read_sql_query("SELECT * from data", connection))
    connection.close()


# Now the list collection is enormous, but since all the dataframes have the same columns in them, I'm just gonna grab one to use as an example for the rest of the exploration.  Specifically, the last one.

# In[ ]:


#just pick the last element of the list, and let's start doing some EDA

df = df_list[-1]

#start with columns, to get an idea of what variables we could have to work with
df.columns


# Right off the bat, the most interesting looking column to me is language.  Knee jerk reaction may be to toss any tweets that aren't in French, but let's see if locations match in a reasonable way.  In other words, if there are Spanish tweets in the Southwest of France about the election, maybe we shouldn't be so quick to drop them.

# In[ ]:


#let's start by taking a look at what regions there are, and what languages

df['lang'].value_counts()


# In[ ]:


df['location'].value_counts()


# This is quite a mess.  There's a jumble of single use locations (including Laos), and a mess of languages as well.  To deal with the languages, we'll use Wikipedia to pick only the top five languages spoken in France.  According to wikipedia, we'll want: French, English, Spanish, German, and Italian.  There's also another category "und" which ranks high up.  We'll nab it too.

# In[ ]:


#we redo the df to only use those languages
langlist = ['fr', 'en', 'und', 'de', 'it', 'es']
df = df[df['lang'].isin(langlist)]
df.head()


# Now our dataframe is a bit more reasonable.  There's still some improvement to be done, specifically on the location column.  We want to thin out the herd a little, too many entries with just 1 hit.  Finding a good cutoff point is really difficult though.  But I'm just going to use this an opportunity to demonstrate some techniques.  Here's what we plan to do:
# 
# 1) Turn entries like "FRANCE", "France", and "france" into "France"
# 
# 2) Turn entries like "Amiens, France" into "Amiens"
# 
# 3) Drop obviously outside France entries

# In[ ]:


#start with number 1

#we'll make a list of the offending names
francelist = ['FRANCE', 'france', 'France', 'France ']

#now simply use the df.replace function from pandas and we're good to go
df['location'] = df['location'].replace(to_replace=francelist, value='France')

#just a quick check to make sure thinks look okay
df['location'].value_counts()


# In[ ]:


#now for goal number 2, my gut instinct is to just use the str.replace function
df['location'] = df['location'].str.replace(', France', '')
df['location'].value_counts()


# Now number three is the toughest.  There's a variety of approaches that can be used here, but the most reasonable one would be to generate a large list of all cities, towns, villages (apparently these fall under the collective name of communes in France) in France and former colonies which can vote.  Some googling shows that there's a module called 'geography' which might work.  Otherwise, this is an enormous task.  Unfortunately, geography is not available to import on Kaggle.  Since that's the case, we'll do some other little fixes and finish up by looking at the five major players (and more specifically the two that are advancing).

# In[ ]:


#paris also suffers from a similar problem to France, let's fix that
#we'll make a list of the offending names
parislist = ['PARIS', 'paris', 'Paris', 'Paris ']

#now simply use the df.replace function from pandas and we're good to go
df['location'] = df['location'].replace(to_replace=parislist, value='Paris')


# In[ ]:


#a decision we can make to deal with some of the trashy locations is to just cull all locations
#that only appear, say, less than 100 times.

#it seems Kaggle can't handle this, but here's how it would be done

#thresholdnum = 100
#value_counts = df['location'].value_counts()
#to_remove = value_counts[value_counts <= thresholdnum].index
#df['location'].replace(to_remove, np.nan, inplace=True)

#df['location'].value_counts()

