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


# # US Demographics EDA
# This EDA uses some of the techniques we learned in class to display some insights from the US Demographics dataset. Some of the interesting insights come from poverty rates in certain counties, rates of public transit ridership in certain counties, percentage of Hispanic residents in certain counties, etc.

# ## **Section 1: Preview**
# ### This section displays some basic information about the dataset (i.e. a preview of the data, the number of rows and columns in the dataset, etc.

# In[ ]:


us_counties = pd.read_csv('/kaggle/input/us-census-demographic-data/acs2017_county_data.csv')
us_counties.head()


# In[ ]:


us_counties.tail()


# In[ ]:


us_counties.shape


# In[ ]:


us_counties.columns


# In[ ]:


us_counties.index


# ## **Section 2: Selecting**
# ### This section will show some specific parts of the dataset instead of the entire dataset.

# Show only counties with populations of more than 100,000 people.

# In[ ]:


us_counties[us_counties.TotalPop >= 100000]


# Show data for Los Angeles County only

# In[ ]:


us_counties.set_index('County').loc['Los Angeles County']


# Show data for the county that is right in the middle in terms of population.

# In[ ]:


us_counties.sort_values('TotalPop', ascending=False).iloc[round(us_counties.shape[0]/2),:]


# ## **Section 3: Summary statistics** 
# ### Display some statistics of the US population

# Display the total population of the US

# In[ ]:


us_counties.loc[:,'TotalPop'].sum()


# Display the average population of each county

# In[ ]:


us_counties.loc[:,'TotalPop'].mean()


# Display the county that has the highest percentage of residents that use public transit

# In[ ]:


us_counties.set_index('Transit').loc[us_counties.loc[:,'Transit'].max()]


# Display the average commute time for cities where at least 30% of residents use public transit

# In[ ]:


us_counties[us_counties.Transit >= 30].loc[:,'MeanCommute'].mean()


# Display the median income of all people in the United States

# In[ ]:


us_counties.loc[:,'Income'].median()


# Display the county that has the lowest poverty rate

# In[ ]:


us_counties.set_index('Poverty').loc[us_counties.loc[:,'Poverty'].min()]


# ## **Section 4: Split-apply-combine** 
# ### Display some characteristics of the US by states

# Display the number of counties in each state

# In[ ]:


us_counties.groupby('State').size()


# Display the average commute time of all counties in each state 
# (each county is weighted equally, not by population)

# In[ ]:


us_counties.groupby('State').mean().loc[:,'MeanCommute']


# Display the average percentage of residents that drive alone to work for all counties in each state 
# (each county is weighted equally, not by population)

# In[ ]:


us_counties.groupby('State').mean().loc[:,'Drive']


# ## **Section 5: Visualization**

# Display a pie chart that shows the 10 most populous states

# In[ ]:


us_counties.groupby('State').sum().loc[:,'TotalPop'].sort_values(ascending=False).head(10).plot.pie()


# Display a bar graph that compares the average percentage of Hispanic residents for all counties 
# in each state

# In[ ]:


us_counties.groupby('State').mean().loc[:,'Hispanic'].plot.bar()

