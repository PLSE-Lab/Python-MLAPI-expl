#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


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
ships = pd.read_csv('../input/ShipLogbookID.csv')


# In[ ]:


ships = pd.read_csv('../input/CLIWOC15.csv')
#read in data


# In[ ]:


ships.head()
#check head


# In[ ]:


temps = ships[['Year', 'Month', 'Day', 'UTC', 'ProbTair']]
#get only the columns we need


# In[ ]:


temps.head()
#look at the head


# In[ ]:


years = temps.pivot_table(index = 'Year', values = 'ProbTair', aggfunc = [np.mean, np.max, np.min])
#pivot table


# In[ ]:


years


# In[ ]:


#There is a problem with 1853
years.ix[1853]


# In[ ]:


years.ix[1853]


# In[ ]:


#to set the new temp, we will replace it with the surrounding data, which in this case is 32.333
# to do this you need to search for the index value, that is .ix call for the dataframe in the previous cell
years.ix[1853].amax
years.ix[1853].amax = 32.222


# In[ ]:


#reduce the year range to drop the large gaps in the data
year2 = years.ix[range(1749, 1856)]


# In[ ]:


#look at the header to see if things are as we expect
year2.head()


# In[ ]:


#import seaborn
import seaborn as sns


# In[ ]:


#pandas organic histogram
year2.plot(kind = 'hist')


# In[ ]:


year2.dropna


# In[ ]:


year2['mean']


# In[ ]:


year2.ix[1853]


# In[ ]:


year2.ix[1853].amax = 32.222


# In[ ]:


year2.ix[1853]


# In[ ]:


year2.plot(kind='hist')


# In[ ]:


len(year2)


# In[ ]:




