#!/usr/bin/env python
# coding: utf-8

# # Not So Common After All
# This project started off as my interest in finding an uncommon baby name. However, after analyzing the data, I realized that common names such as John or Michael aren't that common after all, and that the majority of Americans actually have very unique names. Let's dive right in!

# In[ ]:


# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from collections import defaultdict
sns.set_context('notebook')
from collections import Counter

import copy

from sklearn.preprocessing import MinMaxScaler


# In[ ]:


# Reading data
df = pd.read_csv('../input/NationalNames.csv')


# In[ ]:


df.head()


# In[ ]:


print ('Data year ranges from {} to {}'.format(min(df['Year']),max(df['Year'])))


# The data is listed out by year. I'm more interested in seeing what names are most/least popular by decades. Let's aggregate our data into decades.

# In[ ]:


# Assign decades to Year
df['Decade'] = df['Year'].apply(lambda x: x - (x%10))
df.tail()


# In[ ]:


df_pivot = df.pivot_table(values='Count',index=['Decade','Name','Gender'],aggfunc='sum')
new_df = pd.DataFrame()
new_df['Decade'] = df_pivot.index.get_level_values('Decade')
new_df['Name'] = df_pivot.index.get_level_values('Name')
new_df['Gender'] = df_pivot.index.get_level_values('Gender')
new_df['Count'] = df_pivot.values


# In[ ]:


# This new dataframe contains data aggregated into decades
new_df.head()


# Let's see which names are more rare/popular than others! 
# 
# To define popularity, I will use the MinMaxScale function for names within each decade within gender. This will give us a comparison with how "less" popular a name is compared with a common name like "Michael" which will most likely have a score of 100%.
# 
# This way, we can ask questions like:
# * How common is your name in your birth decade? 
# * Are there people with the same name as you of the opposite gender?
#     * Is my name more common in the opposite gender?

# In[ ]:


decadeList = list(new_df['Decade'].unique())
boys_percentileList = []
girls_percentileList = []

# Making temp copies so I can add a column to the dataframe
boys_df = new_df[new_df['Gender']=='M'].copy()
girls_df = new_df[new_df['Gender']=='F'].copy()

for i in decadeList:
    scaler = MinMaxScaler()
    boys_percentileList.extend(scaler.fit_transform(boys_df[boys_df['Decade']==i][['Count']]))
    girls_percentileList.extend(scaler.fit_transform(girls_df[girls_df['Decade']==i][['Count']]))

boys_df['decade_percentile'] = boys_percentileList
girls_df['decade_percentile'] = girls_percentileList

new_df = boys_df.append(girls_df)

# MinMaxScaler() returns a list of np.arrays, let's change that into a float 
new_df['decade_percentile'] = new_df['decade_percentile'].apply(lambda x: float(x) * 100)
new_df.sort_index(inplace=True)
new_df.head()

# Just cleaning some memory
del boys_df
del girls_df


# Let's see if our popularity metric works! A common name, say, John, should have a score of 100.

# In[ ]:


plt.plot(new_df[(new_df['Name']=='John')&(new_df['Gender']=='M')]['Decade'],
         new_df[(new_df['Name']=='John')&(new_df['Gender']=='M')]['decade_percentile'])


# WHAT? JOHN IS NOT THE MOST COMMON NAME SINCE THE 1920s? WHAT IS THEN???

# In[ ]:


# Listing out the most popular names through the past century
new_df[new_df['decade_percentile']>=99.0]


# Being born in the 1980s, I know my share of Jessica's and Michael's. This metric seems legit with me as the single data point.
# 
# Let's look at the least common names now. I don't want to have 100 kids looking at me at the same time when I call out my future kid's name.

# In[ ]:


# Showing all names with less than 1% popularity
new_df[new_df['decade_percentile'] < 1]


# Obviously my choices are plenty. Too plenty, if you ask me. Let's look at the popularity distribution.

# In[ ]:


plt.figure()
sns.distplot(new_df[(new_df['Gender']=='M')]['decade_percentile'], bins=100)
plt.xlim(xmin=0,xmax=100)
plt.title('Boys Name Popularity Distribution')

plt.figure()
sns.distplot(new_df[(new_df['Gender']=='F')]['decade_percentile'], bins=100)
plt.xlim(xmin=0,xmax=100)
plt.title('Girls Name Popularity Distribution')          

plt.show()


# Wow, seems like for both male and female babies, the 1% least common names make up the majority of the population!
# 
# After taking another look at the least common names in the table above, I think those names are a little too unique for my taste. I need a tool that can help me select names by popularity percentile range, gender, and decade. Why not also add in the ability to select a name based on a starting letter of my choice?

# In[ ]:


def nameFilter(decade,gender,lowerBound,upperBound,startsWith=None):
    '''
        This function helps you find rare/common baby names!
        Inputs:
            decade : integer = Decade as a 4 digit number, e.g. 1980.
            gender : string = Gender as a single letter string, e.g. 'M' for Male
            lowerBound: float = Lower percentage of the names you want to query, e.g. 25 for 25%, NOT 0.25
            upperBound: float = Upper percentage of the names you want to query
            startsWith: str = (Optional) Single letter representing the starting letter of a name
        Returns:
            A dataframe slice fitting your parameters.
    '''
    if upperBound < lowerBound:
        raise ValueError('lowerBound needs to be less than upperBound')
    
    if startsWith != None:
        result_df = new_df[(new_df['Decade'] == decade) &
                           (new_df['Gender'] == gender) &
                           (new_df['decade_percentile'] >= lowerBound) &
                           (new_df['decade_percentile'] <= upperBound) &
                           (new_df['Name'].str[0]==startsWith.upper())
                          ]
    else:
        result_df = new_df[(new_df['Decade'] == decade) &
                           (new_df['Gender'] == gender) &
                           (new_df['decade_percentile'] >= lowerBound) &
                           (new_df['decade_percentile'] <= upperBound) 
                          ]
    return result_df


# Say I want to know which baby boy names were the most commonly picked in the 1980's that starts with the letter 'C', defined by having a popularity score of 50 or more.

# In[ ]:


nameFilter(decade=1980, gender='M', lowerBound=50, upperBound=100, startsWith='C')


# The filter came back with a surprisingly short list. This result is in line with the frequency charts in the middle of the notebook. Seems like Americans are very creative in giving their kids unique names!
