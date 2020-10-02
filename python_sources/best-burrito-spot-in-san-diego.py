#!/usr/bin/env python
# coding: utf-8

# This notebook is designed to find the best burrito spot in San Diego based on the features in the description. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/burritodata_092616.csv')
df = df.fillna(df.mean())
df.head()


# After loading the dataset, I wanted to extract the *10 dimensions* mentioned in the description for my analysis.

# In[ ]:


columnList = ['Location', 'Burrito', 'Cost', 'Volume', 'Tortilla', 'Temp', 'Meat', 'Fillings', 'Meat:filling', 'Uniformity', 'Salsa', 'Synergy', 'Wrap', 'overall']
dfNew = df[columnList]
dfNew.head()


# Once I had my new dataset, I wanted to see what features affect the *overall* rating of a burrito. Below is it's representation:

# In[ ]:


corr = dfNew.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
ax = sns.heatmap(corr, mask=mask, cmap = cmap, linewidths= 0.25)
ax = ax.set_title('Correlation Heatmap')


# Great! There are a lot of highly correlated features with the *overall* feature. Let us see how they stack up:

# In[ ]:


corr = dfNew.corr()['overall'].reset_index()
corr = corr.sort_values(by = 'overall')[:(len(corr)-1)]
ax = sns.barplot(y = corr['index'], x = corr['overall'], palette='Greens')
ax.set_xlabel('Correlation')
ax.set_ylabel('Features')
ax = ax.set_title('Correlation between Overall Score and features')


# Turns out *Synergy* feature has the highest pearson's correlation value. I went on to the GitHub page to understand what Synergy meant in the Burrito context and I got to know that it is the coming together of all the components for making that perfect Burrito.
# A burrito is rated on the quality of the tortilla, meat, the meat to fillings ratio etc. and when they all come together, it turns from a good burrito to an unforgettable experience. 

# In[ ]:


Now, to analyse about the best locations, I first wanted to see the count of reviews for each location and then consider only those places that had at least 5 reviews.


# In[ ]:


x = df.groupby('Location').size().reset_index()
temp = x.sort_values(ascending=False, by=0)
temp


# There are only 9 places that have more than 4 reviews, so I will be performing my analysis on them.

# In[ ]:


temp = temp[:9]
y = pd.DataFrame()
for name in temp['Location']:
    y = y.append(df[(df['Location']==name)])
x = y.groupby('Location').mean().reset_index()
x = x.sort_values(by = 'overall', ascending=False)
plt.rcParams['figure.figsize'] = (15.0, 4.0)
ax = sns.barplot(x = x['Location'], y = x['overall'], palette='Greens')
ax.set_ylabel('Rating')
ax = ax.set_title('Overall Ranking for Locations with 5 or more reviews')


# Taco Stand and California Burritos are neck and neck for the best Burrito whereas Lucha Libre North Park and Primos Mexican Food aren't doing that well on the ratings. Let us explore a bit more:

# Now, I wanted to find the best burrito place that gives me *best bang for the buck*. There are two ways to approach this problem:<br/>
# 1. Find the best ratings and then sort w.r.t lowest price<br/>
# 2. Find the lowest price and then sort w.r.t highest rating

# In[ ]:


x = x.sort_values(by = ['overall','Cost'], ascending=[False,True])
x[['Location', 'overall', 'Cost']]


# In[ ]:


temp = y.groupby('Location').mean().reset_index()
x = temp.sort_values(by = ['Cost','overall'], ascending=[True,False])
x[['Location', 'Cost', 'overall']]


# Taco Stand is a better option if the priority is for a better overall burrito that is just about a dollar expensive than California Burrito.

# For the next part of the analysis, I wanted to see if the locations were consistent based on individual features like Tortilla, Meat quality, Salsa etc. Below are some plots that are self explanatory:

# In[ ]:


plt.rcParams['figure.figsize'] = (6.0, 4.0)
x = temp.sort_values(by = 'Tortilla', ascending=False)
x = x[['Location', 'Tortilla']]
x = x.iloc[[0,-1]]
ax = sns.barplot(x = 'Location', y = 'Tortilla', data = x)
ax.set_ylabel('Tortilla Rating')
ax = ax.set_title('Best/Worst Tortilla Ratings')


# In[ ]:


x = temp.sort_values(by = 'Fillings', ascending=False)
x = x[['Location', 'Fillings']]
x = x.iloc[[0,-1]]
ax = sns.barplot(x = 'Location', y = 'Fillings', data = x)
ax.set_ylabel('Filling Rating')
ax = ax.set_title('Best/Worst Filling Ratings')


# In[ ]:


x = temp.sort_values(by = 'Meat:filling', ascending=False)
x = x[['Location', 'Meat:filling']]
x = x.iloc[[0,-1]]
ax = sns.barplot(x = 'Location', y = 'Meat:filling', data = x)
ax.set_ylabel('Meat:Filling Rating')
ax = ax.set_title('Best/Worst Meat:Filling Ratings')


# In[ ]:


x = temp.sort_values(by = 'Meat', ascending=False)
x = x[['Location', 'Meat']]
x = x.iloc[[0,-1]]
ax = sns.barplot(x = 'Location', y = 'Meat', data = x)
ax.set_ylabel('Meat Rating')
ax = ax.set_title('Best/Worst Meat Ratings')


# In[ ]:


x = temp.sort_values(by = 'Salsa', ascending=False)
x = x[['Location', 'Salsa']]
x = x.iloc[[0,-1]]
ax = sns.barplot(x = 'Location', y = 'Salsa', data = x)
ax.set_ylabel('Salsa Rating')
ax = ax.set_title('Best/Worst Salsa Ratings')


# As we can see that the locations with the best ingredients aren't consistent. Some place has better tortilla and some other place has the best meat or salsa. But, the poor rating locations are consistent.

# For the next part, I wanted to finally plot the distributions of features for Taco Stand and California Burrito(best ratings) and Lucha Libre, Los Primos(worst ratings). This would give us a better picture of what to expect in these locations.

# In[ ]:


plt.rcParams['figure.figsize'] = (9.0, 4.0)
x = y[(y['Location']=='California Burritos')]
ax = sns.distplot(x['Tortilla'], color='red', hist=False, label='California Burritos')
x = y[(y['Location']=='Taco Stand')]
ax = sns.distplot(x['Tortilla'], color='blue', hist=False, label='Taco Stand')
x = y[(y['Location']=="Lucha Libre North Park")]
ax = sns.distplot(x['Tortilla'], color='green', hist=False, label='Lucha Libre North Park')
x = y[(y['Location']=="Los Primos Mexican Food")]
ax = sns.distplot(x['Tortilla'], color='black', hist=False, label='Los Primos Mexican Food')
ax.set_ylabel('Density')
ax = ax.set_xlabel('Tortilla Rating')


# In[ ]:


x = y[(y['Location']=='California Burritos')]
ax = sns.distplot(x['Meat:filling'], color='red', hist=False, label='California Burritos')
x = y[(y['Location']=='Taco Stand')]
ax = sns.distplot(x['Meat:filling'], color='blue', hist=False, label='Taco Stand')
x = y[(y['Location']=="Lucha Libre North Park")]
ax = sns.distplot(x['Meat:filling'], color='green', hist=False, label='Lucha Libre North Park')
x = y[(y['Location']=="Los Primos Mexican Food")]
ax = sns.distplot(x['Meat:filling'], color='black', hist=False, label='Los Primos Mexican Food')
ax.set_ylabel('Density')
ax = ax.set_xlabel('Meat:Filling Rating')


# Aha! California Burritos has a better Meat to Filling Ratio than Taco Stand! Something to cheer about!

# In[ ]:


plt.rcParams['figure.figsize'] = (9.0, 4.0)
x = y[(y['Location']=='California Burritos')]
ax = sns.distplot(x['Fillings'], color='red', hist=False, label='California Burritos')
x = y[(y['Location']=='Taco Stand')]
ax = sns.distplot(x['Fillings'], color='blue', hist=False, label='Taco Stand')
x = y[(y['Location']=="Lucha Libre North Park")]
ax = sns.distplot(x['Fillings'], color='green', hist=False, label='Lucha Libre North Park')
x = y[(y['Location']=="Los Primos Mexican Food")]
ax = sns.distplot(x['Fillings'], color='black', hist=False, label='Los Primos Mexican Food')
ax.set_ylabel('Density')
ax = ax.set_xlabel('Fillings Rating')


# In[ ]:


x = y[(y['Location']=='California Burritos')]
ax = sns.distplot(x['Salsa'], color='red', hist=False, label='California Burritos')
x = y[(y['Location']=='Taco Stand')]
ax = sns.distplot(x['Salsa'], color='blue', hist=False, label='Taco Stand')
x = y[(y['Location']=="Lucha Libre North Park")]
ax = sns.distplot(x['Salsa'], color='green', hist=False, label='Lucha Libre North Park')
x = y[(y['Location']=="Los Primos Mexican Food")]
ax = sns.distplot(x['Salsa'], color='black', hist=False, label='Los Primos Mexican Food')
ax.set_ylabel('Density')
ax = ax.set_xlabel('Salsa Rating')


# In[ ]:


x = y[(y['Location']=='California Burritos')]
ax = sns.distplot(x['Synergy'], color='red', hist=False, label='California Burritos')
x = y[(y['Location']=='Taco Stand')]
ax = sns.distplot(x['Synergy'], color='blue', hist=False, label='Taco Stand')
x = y[(y['Location']=="Lucha Libre North Park")]
ax = sns.distplot(x['Synergy'], color='green', hist=False, label='Lucha Libre North Park')
x = y[(y['Location']=="Los Primos Mexican Food")]
ax = sns.distplot(x['Synergy'], color='black', hist=False, label='Los Primos Mexican Food')
ax.set_ylabel('Density')
ax = ax.set_xlabel('Synergy Rating')


# To end the analysis, it is the coming together of all the features that counts i.e. the Synergy feature. Some place may have the better tortilla or meat but it is the coherent nature of all the flavours and ingredients coming together to form a beautiful music like harmony and the last plot approves it for Taco Stand.
# Happy Eating!
