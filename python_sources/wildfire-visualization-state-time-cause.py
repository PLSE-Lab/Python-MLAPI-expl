#!/usr/bin/env python
# coding: utf-8

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


#  First, I want to see how the number of wildfires may vary up to year/date. We can take a look at the following bar graph and at 2006 there are too many wildfires compared to other years. 

# In[ ]:


# WildFire in US
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

conn = sqlite3.connect('../input/FPA_FOD_20170508.sqlite')


# In[ ]:


df = pd.read_sql_query("SELECT * FROM 'Fires'", conn)
df.head()


# In[ ]:


#Creating a boolean w T/Fs
ca_only = df['STATE'] == 'CA'

#Slicing dataset into only fires within CA
ca_only_ds = df[ca_only]



# In[ ]:


list(ca_only_ds)


# In[ ]:


# 
fire_year = df.groupby('FIRE_YEAR').size()
plt.bar(range(len(fire_year)), fire_year.values, width = 0.5)
plt.xticks(range(len(fire_year)), fire_year.index, rotation = 50)
plt.show()


# Moreover, it also varies up to date. We can see some local maximums on April and July and local minimum on May and June. 

# In[ ]:


fire_date = df.groupby('DISCOVERY_DOY').size()
plt.scatter(fire_date.index, fire_date.values)
plt.show()


# Lastly, I want to see the relation between the cause and the fire year.  Interestingly, there are a large number of wildfires in 2006 because of Debris Burning. 

# In[ ]:


sns.heatmap(pd.crosstab(df.FIRE_YEAR, df.STAT_CAUSE_DESCR))
plt.show()


# Lastly, You may vary the years and the causes and check distributions of the number of wildfires on each state in the following Tableau dashboard. Another interesting fact we observe by going through the following visualization is that in 2006 there are a large number of wildfires in Texas due to debris burning. 
# 
# https://us-west-2b.online.tableau.com/t/jungajunga/views/WildfiresinUS/Dashboard1?:embed=y&:showAppBanner=false&:showShareOptions=true&:display_count=no&:showVizHome=no
# 
# (Please let me know if the link does not work.)
