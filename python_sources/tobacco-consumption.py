#!/usr/bin/env python
# coding: utf-8

# #Tobacco Consumption
# The tobacco consumption data set contains the consumption of tobacco of people living in the Unites States (in percentage), with information including the percentage of how many people smoke everyday, smoke occasionally, and other information such as, people who used to smoke, and people who never smoked in a yearly basis. The total states that participated in this data set is 56 (50 states and 6 unincorporated).
# 
# 1. How does each tobacco consumption column correlate with each other?
# 2. What is the top 10 highest and top 10 lowest average smoke consumption per state?
# 3. Based on the results from number 2, what are the common behaviors a between the top 5 highest and top 5 lowest states over the years?

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


tobacco = pd.read_csv('../input/tobacco.csv')
tobacco.set_index('Year', inplace = True)
tobacco


# In[ ]:


tobacco.dtypes


# In[ ]:


tobacco['Smoke everyday'] = tobacco['Smoke everyday'].str[:-1].astype(float)
tobacco['Smoke some days'] = tobacco['Smoke some days'].str[:-1].astype(float)
tobacco['Former smoker'] = tobacco['Former smoker'].str[:-1].astype(float)
tobacco['Never smoked'] = tobacco['Never smoked'].str[:-1].astype(float)


# In[ ]:


tobacco.dtypes


# In[ ]:


tobacco


# #1. How does each tobacco consumption column correlate with each other? Particularly, how does smoking everyday correlate to each column?

# In[ ]:


tobacco.corr()


# In[ ]:


tobacco[['Smoke everyday', 'Never smoked']].plot(kind = 'scatter', x = 0, y = 1)
plt.show()


# The correlation of daily smokers to each column, have either no correlation or a negative correlation. The notable correlations are those who are daily smokers compared to those who never smoked has a high negative correlation. It maybe because the population increases, the sample size increased as well, meaning those who have never smoked increases while daily smokers, have decreased over the years.

# #2. What is the top 10 highest and top 10 lowest average smoke consumption per state?

# In[ ]:


avg_tobacco = tobacco.groupby('State')[['Smoke everyday', 'Smoke some days', 'Former smoker', 'Never smoked']].mean()
avg_tobacco


# In[ ]:


avg_tobacco[['Smoke everyday', 'Smoke some days']].sort_values(['Smoke everyday', 'Smoke some days'], ascending = True).tail(10).plot(kind = 'barh')
plt.show()


# In[ ]:


avg_tobacco[['Smoke everyday', 'Smoke some days']].sort_values(['Smoke everyday', 'Smoke some days'], ascending = False).tail(10).plot(kind = 'barh')
plt.show()


# The first horizontal bar graph shows the ten states with the highest average daily smoking consumption. With Nevada being the 10th and Kentucky being the 1st, the highest daily smoking states ranges from 19 to 25. The second horizontal bar graphs shows the top ten states with the lowest average daily smoking consumption. With Virgin Islands being the 1st and Maryland being the 10th the lowest daily smoking states ranges from 5 to 15. Ultimately, it can be assumed that the number of non-smokers per state is higher than the people who smokes, or used to smoke.
# 

# #3. Based on the results from number 2, what are the common behaviors a between the top 5 highest and top 5 lowest states over the years?

# #Top 5 states with the highest average tobacco consumption

# In[ ]:


tobacco[(tobacco.State == 'Kentucky')].sort_index().plot()
plt.show()


# In[ ]:


tobacco[(tobacco.State == 'Guam')].sort_index().plot()
plt.show()


# In[ ]:


tobacco[(tobacco.State == 'West Virginia')].sort_index().plot()
plt.show()


# In[ ]:


tobacco[(tobacco.State == 'Tennessee')].sort_index().plot()
plt.show()


# In[ ]:


tobacco[(tobacco.State == 'Missouri')].sort_index().plot()
plt.show()


# #Top 5 states with the lowest average tobacco consumption

# In[ ]:


tobacco[tobacco.State == 'Virgin Islands'].sort_index().plot()
plt.show()


# In[ ]:


tobacco[tobacco.State == 'Puerto Rico'].sort_index().plot()
plt.show()


# In[ ]:


tobacco[tobacco.State == 'Utah'].sort_index().plot()
plt.show()


# In[ ]:


tobacco[tobacco.State == 'California'].sort_index().plot()
plt.show()


# In[ ]:


tobacco[tobacco.State == 'District of Columbia'].sort_index().plot()
plt.show()


# The results of the line graph in each highest average tobacco consumption, has a pattern that can be seen in people who smokes everyday, and people who used to smoke. With the exception of Guam, all states had an intersection between the rates of people who smokes everyday, and people who used to smoke. As the the number of people who smokes everyday decreases, the number of former smokers increases. Based on the graphs, it is hard to predict if the number of daily smokers are decreasing every year, but if it does, the number of former smokers and possibly the number of occasional smokers will increase. Additionally, people who never smoked never decreased throughout the years.
# 
# The line graphs of each state that has the lowest average tobacco consumption, has particular patterns that can be followed except that the rates of people who smokes daily, and former smokers are generally lower compared to the states that has the highest average tobacco consumption. The visible changes that can be seen from the graphs is similar to the changes discussed above.
