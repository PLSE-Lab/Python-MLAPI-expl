#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
import warnings
import string
import time
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['font.size'] = 15


# In[22]:


data_2015 = pd.read_csv('../input/2015.csv')
data_2016 = pd.read_csv('../input/2016.csv')
data_2017 = pd.read_csv('../input/2017.csv')
data_2017.head()


# ## Happiness Score Distribution

# In[25]:


sns.distplot(data_2015['Happiness Score'], rug=True, rug_kws={"color": "g"},
                   kde_kws={"color": "k", "lw": 3, "label": "KDE"},
                  hist_kws={"histtype": "step", "linewidth": 3,
                            "alpha": 1, "color": "g"})
plt.ylabel("Percentile")
plt.title("Happiness Score 2015")
plt.show()


# In[24]:


sns.distplot(data_2016['Happiness Score'], rug=True, rug_kws={"color": "g"},
                   kde_kws={"color": "k", "lw": 3, "label": "KDE"},
                  hist_kws={"histtype": "step", "linewidth": 3,
                            "alpha": 1, "color": "g"})
plt.ylabel("Percentile")
plt.title("Happiness Score 2016")
plt.show()


# In[23]:


sns.distplot(data_2017['Happiness.Score'], rug=True, rug_kws={"color": "g"},
                   kde_kws={"color": "k", "lw": 3, "label": "KDE"},
                  hist_kws={"histtype": "step", "linewidth": 3,
                            "alpha": 1, "color": "g"})
plt.ylabel("Percentile")
plt.title("Happiness Score 2017")
plt.show()


# ## Happiness Trend of Top 3 countries

# In[60]:


countries = ['Switzerland', 'Iceland', 'Denmark', 'Norway']

for country in countries:
    happiness_score = [data_2015[data_2015['Country'] == country]['Happiness Score'].values,
                      data_2016[data_2016['Country'] == country]['Happiness Score'].values,
                      data_2017[data_2017['Country'] == country]['Happiness.Score'].values]
    #print(happiness_score)
    years = [2015,2016,2017]
    
    ax = sns.pointplot(x=years, y=happiness_score)
    plt.xlabel("Years")
    plt.ylabel("Happiness Score")
    plt.title("Happiness Trend of " + country)
    plt.show()


# Happiness score has decreased for Denmark, Switzerland and Iceland. But for **Norway**, it has increased significantly.

# In[76]:


## Countries whose happiness score increased/decersed during 3 year span

countries = data_2017['Country'].values
countries_with_increasing_happiness = []
countries_with_decreasing_happiness = []

for country in countries:
    happiness_score = [data_2015[data_2015['Country'] == country]['Happiness Score'].values,
                      data_2016[data_2016['Country'] == country]['Happiness Score'].values,
                      data_2017[data_2017['Country'] == country]['Happiness.Score'].values]
    
    if ((happiness_score[0] > happiness_score[1]) and (happiness_score[1] > happiness_score[2])):
        countries_with_decreasing_happiness.append(country)
        
    if ((happiness_score[0] < happiness_score[1]) and (happiness_score[1] < happiness_score[2])):
        countries_with_increasing_happiness.append(country)
    
print("Countries with Increasing happiness score are :\n" + str(len(countries_with_increasing_happiness)))
print(countries_with_increasing_happiness)


# In[77]:


print("Countries with decreasing happiness score are :\n" + str(len(countries_with_decreasing_happiness)))
print(countries_with_decreasing_happiness)


# There are **52** countries whose happiness score is increasing year after year and **51** countries whose happiness score has been decreased.

# In[ ]:




