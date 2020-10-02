#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import math
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


import fuzzywuzzy
from fuzzywuzzy import process

from wordcloud import WordCloud, STOPWORDS

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", "../input/"]).decode("utf8").strip()
# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)


# In[2]:


df = pd.read_csv('../input/FastFoodRestaurants.csv')


# In[3]:


df.head()


# In[4]:


# from Rachel Tatman tutorial
# https://www.kaggle.com/rtatman/data-cleaning-challenge-inconsistent-data-entry?scriptVersionId=3012975

# function to replace rows in the provided column of the provided dataframe
# that match the provided string above the provided ratio with the provided string
def replace_matches_in_column(df, column, string_to_match, min_ratio = 90):
    # get a list of unique strings
    strings = df[column].unique()
    
    # get the top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings, 
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    # only get matches with a ratio > 90
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

    # get the rows of all the close matches in our dataframe
    rows_with_matches = df[column].isin(close_matches)
    print(len(close_matches))
    # replace all rows with close matches with the input matches 
    df.loc[rows_with_matches, column] = string_to_match
    
    # let us know the function's done
    print("All done!")
    return df.copy() 


# In[6]:


sorted(df.name.unique())


# In[7]:


sorted(df.city.unique())


# In[8]:


sorted(df.country.unique())


# In[9]:


df['cleanname'] = df.name.str.lower().str.strip()


# In[10]:


df.head()


# In[64]:


names = sorted(df.cleanname.unique())
names


# In[65]:


phut = fuzzywuzzy.process.extract("pizza hut", names, limit=15, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# take a look at them
phut


# In[66]:


names = sorted(df.cleanname.unique())

mac = fuzzywuzzy.process.extract("mcdonald's", names, limit=15, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# take a look at them
mac


# In[15]:


replace_matches_in_column(df,'cleanname',"mcdonald's", .57)


# In[19]:


sorted(df.cleanname.unique())


# In[67]:


names = sorted(df.cleanname.unique())

dom = fuzzywuzzy.process.extract("domino's pizza", names, limit=15, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# take a look at them
dom


# In[24]:


replace_matches_in_column(df,'cleanname',"domino's pizza", .88)


# In[68]:


sub = fuzzywuzzy.process.extract("subway", names, limit=15, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# take a look at them
sub


# In[69]:


bking = fuzzywuzzy.process.extract("burger king", names, limit=15, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# take a look at them
bking


# In[70]:


nnout = fuzzywuzzy.process.extract("in-n-out burger", names, limit=15, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# take a look at them

nnout


# In[71]:


names = sorted(df.cleanname.unique())

kfc = fuzzywuzzy.process.extract("kfc", names, limit=15, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# take a look at them
kfc


# In[72]:


names = sorted(df.cleanname.unique())

fuzzywuzzy.process.extract("popeye's", names, limit=15, scorer=fuzzywuzzy.fuzz.token_sort_ratio)


# In[30]:



names = sorted(df.cleanname.unique())

kfc = fuzzywuzzy.process.extract('kentucky fried chicken', names, limit=15, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# take a look at them
kfc


# In[29]:


df = replace_matches_in_column(df,'cleanname','kentucky fried chicken', .91)


# In[26]:


df = replace_matches_in_column(df,'cleanname',"kfc", .37)


# In[31]:


df.cleanname.value_counts()


# In[46]:


df.head()


# In[59]:


df.loc[df.cleanname.str.startswith('kfc'), 'cleanname'] = 'kentucky fried chicken'
df.loc[df.cleanname.str.startswith("mcdonald's"), 'cleanname'] = "mcdonald's"
df.loc[df.cleanname.str.startswith('burger king salou'), 'cleanname'] = 'burger king'
df.loc[df.cleanname.str.startswith("popeye"), 'cleanname'] = "popeye's"
df.loc[df.cleanname.str.startswith("pizza hut"), 'cleanname'] = "pizza hut"
df.loc[df.cleanname.str.startswith("subway"), 'cleanname'] = "subway"
    


# In[50]:


sum(df.cleanname.str.startswith('kfc'))


# In[63]:


df.cleanname.value_counts().sort_index()


# In[61]:


sorted(df.cleanname.unique())


# In[73]:


myfavorits = ['kentucky fried chicken',"mcdonald's",'burger king',"popeye's","pizza hut","subway","domino's pizza"]


# In[77]:


dfav = df[df.cleanname.isin(myfavorits)]


# In[78]:


len(df)


# In[79]:


len(dfav)


# In[80]:


dfav.cleanname.value_counts()


# In[82]:


dfav.head()


# In[84]:


sns.countplot(x = 'cleanname', data = dfav)


# In[85]:


sns.countplot(x = 'province', data = dfav)


# In[87]:


sns.countplot(x = 'province', y = 'cleanname' data = dfav)


# In[92]:


sns.lmplot(x='longitude', y='latitude', hue='cleanname', 
           data=dfav, 
           fit_reg=False, scatter_kws={'alpha':0.2})


# In[95]:


sns.boxplot(data = dfav, x = 'cleanname', y ='longitude',hue = 'cleanname')#, fit_reg= False, markers = ['x', 'o'])


# In[ ]:




