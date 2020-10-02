#!/usr/bin/env python
# coding: utf-8

# **This is a notebook consisting of various plots to describe the disparities of Top 500 Indian cities.I am a beginner so any advise or suggestion on improvement is most welcome**

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


# *Importing modules*

# In[ ]:


import os
import pandas as pd
import numpy as np
from pandas import DataFrame,Series
from collections import Counter
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


f = pd.read_csv('../input/cities_r2.csv')
data=DataFrame(f)
print(data.head())


# **Most populated cities**

# In[ ]:


most_populated_cities=data.sort(['population_total'], ascending=[0] )[:5]
most_populated_cities=pd.crosstab(most_populated_cities.name_of_city,most_populated_cities.population_total)
most_populated_cities.plot(kind='bar')


# *States with most cities in Top 500 cities*

# In[ ]:


top_states=Counter(data['state_name'])
top_States=sorted(top_states.items(), key=lambda x: x[1])
top_States=DataFrame(top_States[-5:])
States=list(top_States[0])
print(States)
top_States=DataFrame({"States":top_States[0],"Number_of_cities":top_States[1]})
x=[1,2,3,4,5]
y=top_States['Number_of_cities']
y=list(y)
print(y)
plt.bar(x,y)
plt.xticks(x,States,rotation='vertical')
plt.margins(0.1)
plt.subplots_adjust(bottom=0.15)


# *SEX Ratio comparison*

# In[ ]:


sex_ratio=data.sort(['sex_ratio'],ascending=[0])[:15]
print(sex_ratio['state_name'])


# In[ ]:


literacy_rate=data[data['effective_literacy_rate_total']>90]
literacy_rate_states=literacy_rate['state_name']
literacy_rate_states=Counter(literacy_rate_states)
plt.figure(figsize=(12,7))
plt.bar(range(len(literacy_rate_states)), literacy_rate_states.values(), align='center',color='g')
plt.xticks(range(len(literacy_rate_states)), literacy_rate_states.keys(),rotation='vertical')
plt.show()


# **Graduates comparison**

# In[ ]:


graduates=data.sort(['total_graduates'],ascending=[0])[:15]
plt.figure(figsize=(12,7))
plt.bar(range(len(graduates['total_graduates'])), graduates['total_graduates'], align='center',color='r')
graduates_states=list(graduates['state_name'])
plt.xticks(range(len(graduates_states)), graduates_states,rotation='vertical')


# In[ ]:


merged_graduates=graduates[['total_graduates','male_graduates','female_graduates']].copy()
plt.figure(figsize=(20,10))
merged_graduates.plot(kind='bar')
plt.xticks(range(len(graduates_states)), graduates_states,rotation='vertical')


# In[ ]:




