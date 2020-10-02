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


# Solutions for Tasks on Maryland Traffic Violations Data Set
# 
# To start, let's load the data.  In this case, there are quite a few columns and not all of them will be of interest for the task solutions.

# In[ ]:


df = pd.read_csv(r'/kaggle/input/traffic-violations-in-maryland-county/Traffic_Violations.csv')
df.head()


# In[ ]:


df.columns


# To try and get to the answer, let's take that dataset and pull a subset of the data for which we are interested.  In the next cell, lines 2-4 are recoding the Yes/No data in the Alchol, Belts, and Fatal columns to be binary, which allows for the statistics.  For this analysis, the index columns does not really matter, but we will use it later.

# In[ ]:


sub_df = pd.DataFrame(zip(df['Article'],df['Alcohol'],df['Belts'],df['Fatal']), columns=['violation','alcohol','belts','fatal'])
sub_df['alcohol'] = sub_df.alcohol.eq('Yes').mul(1)
sub_df['belts'] = sub_df.belts.eq('Yes').mul(1)
sub_df['fatal'] = sub_df.fatal.eq('Yes').mul(1)
sub_df.set_index('violation').describe()


# So, of all the violations in the data, 0.171% are associated with alcohol, 0.344% are associated with no seat belt, and 0.216% resulted in a fatality. This is because we recoded these columns to be binary, so the mean of the column will reflect the percentage that involved the associated condition.
# 
# Digging a little deeper into the data, we can correlate the violation type and produce a pivot table to see what types of violation types were associated with each of these. 
# 
# *Note: This step was also done with the article above, but was not as useful as the article is a higher level metric (e.g. Electricity vs. Alternating Current or Direct Current).*

# In[ ]:


sub_df1 = pd.DataFrame(zip(df['Violation Type'],df['Alcohol'],df['Belts'],df['Fatal']), columns=['violation','alcohol','belts','fatal'])
sub_df1['alcohol'] = sub_df1.alcohol.eq('Yes').mul(1)
sub_df1['belts'] = sub_df1.belts.eq('Yes').mul(1)
sub_df1['fatal'] = sub_df1.fatal.eq('Yes').mul(1)
table1 = pd.pivot_table(sub_df1, values=['alcohol','belts','fatal'], columns='violation', aggfunc=np.mean)
table1


# OK, so now we can see that Alcohol violations generated citations 18 (0.000190/0.003427) times more often than they generated warnings.  This is much higher than the 1.6 times for Seat Belts (0.043493/0.026080) or 2.2 (0.000310/0.000142) for Fatal Crashes.  As a result, it would seem that Maryland Police are more strict on alcohol violations than for the other two. 
# 
# As seen below, we can rack and stack these to look at articles/types/violation combinations and see where the three correlate.  While this doesn't display the whole table, we can clearly see that for violation code 11-391.41(a) all three factors contributed to those citations.

# In[ ]:


sub_df1 = pd.DataFrame(zip(df['Date Of Stop'],df['Article'],df['Violation Type'],df['Charge'],df['Alcohol'],df['Belts'],df['Fatal']), columns=['date','article','type','violation','alcohol','belts','fatal'])
sub_df1['alcohol'] = sub_df1.alcohol.eq('Yes').mul(1)
sub_df1['belts'] = sub_df1.belts.eq('Yes').mul(1)
sub_df1['fatal'] = sub_df1.fatal.eq('Yes').mul(1)
sub_df1 = sub_df1[(sub_df1.alcohol > 0) | (sub_df1.belts > 0) | (sub_df1.fatal > 0)]
table1 = pd.pivot_table(sub_df1, values=['alcohol','belts','fatal'], index=['article','type','violation'], aggfunc=np.sum)
table1


# We can also accumulate discrete bins by violation to see the largest magnitude of violations.  For alcohol, the larges number of violations was 674. In contrast, seat belts had a much higher violation rate for the principal code at 6548. Fatal violations were much lower at 36. 

# In[ ]:


table1 = pd.DataFrame(table1, columns=['alcohol','belts','fatal'])
alc_df = table1[table1['alcohol']!=0]
alc_df = alc_df.sort_values('alcohol', ascending=False)
alc_df = pd.DataFrame(alc_df.alcohol,columns=['alcohol'])
alc_df.head(30)


# In[ ]:


belts_df = table1[table1['belts']!=0]
belts_df = belts_df.sort_values('belts', ascending=False)
belts_df = pd.DataFrame(belts_df.belts,columns=['belts'])
belts_df.head(30)


# In[ ]:


fatal_df = table1[table1['fatal']!=0]
fatal_df = fatal_df.sort_values('fatal', ascending=False)
fatal_df = pd.DataFrame(fatal_df.fatal,columns=['fatal'])
fatal_df.head(30)


# One clear conclusion here is that, despite the obvious presence of seat belt enforcement measures on the streets of Maryland (which I see everyday), there is still a surprisingly large number of violations of the seat belt code.  This might motivate local and state governments to put more resources toward this or work with vehicle manufacturers to develop methods to reduce such violations.
