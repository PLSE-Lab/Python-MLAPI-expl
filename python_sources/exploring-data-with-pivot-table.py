#!/usr/bin/env python
# coding: utf-8

# Hi guys!
# 
# In this notebook i will do some data analysis on nyc-jobs info using pivot tables
# 
# the goal here is explore our data and the pivot_tables function, the information will be presented only by tables, without graphs.
# 
# that's all folks, lets code.
# 
# 
# 
# ![](http://www.quickmeme.com/img/59/595313db37b222d68fb731c92b52ab466313d47c0677d2313012d7841da49ed9.jpg)

# First of all, lets import our libraries and the dataset, using .head(3) only to see what we have in the table.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

dataset =  pd.read_csv("../input/nyc-jobs.csv")
dataset.head(3)


# looking the data i can see some points that it will be fast to analyse with pivot_tables, soo lets analyse these points:
# 
# **Proportion of External/Internal Jobs by Agency:**
# Explore if a agency priorize more the internal recruitment than the external. My point here is that agencies that priorize the internal recruitment than external appreciate more they employeers.
# 
# **Mean of minimum wage by business title:**
# Know the minimum expected wage for each business title
# 
# **Division Units jobs by agencies:**
# See how many jobs we have for each division unit, by agency.

# In[ ]:


#Proportion of External/Internal Jobs by Agency
postingtype_rate = dataset.pivot_table(values = 'Job ID', index='Agency', columns ='Posting Type', aggfunc='count')
postingtype_rate['Racional'] = postingtype_rate['External']/(postingtype_rate['External'] + postingtype_rate['Internal'])
postingtype_rate.sort_values(by='Racional', ascending = False).head(5)


# In the table we can see that our racional between External/Internal jobs almost never pass 0.5, here we have only **Business Integrity Comission** with a rational of 0.6
# 
# So agencies are not priorizing external than internal, or they are equal or we have more internal opportunities than external. This is good.

# In[ ]:


#Mean of minimum wage by business title (top 10)
pd.options.display.float_format = '{:,.2f}'.format
wages_mean = dataset.pivot_table(values = 'Salary Range From', index='Business Title', aggfunc='mean')
wages_mean.rename(columns={'Salary Range From': 'Mean salary'}, inplace=True)
wages_mean['Mean salary monthly'] = wages_mean['Mean salary']/12
wages_mean.sort_values(by = 'Mean salary', ascending=False).head(10)


# Ok, 60% of jobs in our top 10 is to work with government... I strongly disagree with the ridiculous gap between public vs private carrer wage, doesn't make any sense a deputy commissioner earn more than 10k month, considering all the perks paid by the government.

# In[ ]:


#Division Units jobs by agencies
dataset.pivot_table(values='Job ID', index = 'Agency', columns = 'Division/Work Unit', aggfunc='count', fill_value=0, margins=True).sort_values(by='All', ascending=False)


# I dont have anything to point about this last table, it's just to a better visualization of the information
# 
# 
# 
# 
# thanks for reading :)

# In[ ]:





# 
