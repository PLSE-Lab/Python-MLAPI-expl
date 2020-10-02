#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')

# Any results you write to the current directory are saved as output.


# In[ ]:


complaints = pd.read_csv("../input/consumer_complaints.csv", low_memory=False)


# # Introduction
# 
# Since this data is focused on consumer complaints, we'd like to know if there are companies which
# receive far more complaints than you'd expect based on their size. 
# 
# First we'll simply identify outliers, but after that we'd like to use some more sophisticated 
# metrics to understand which companies are receiving "worse" complaints than others.

# In[ ]:





# ### Complaints by Company 

# In[ ]:


# The amount of complaints each company has
grouped_by_company = complaints.company.value_counts()

# Check out the distribution
grouped_by_company.plot(kind='hist')


# So it turns out that most companys have 1 complaint. This shouldn't be very surprising, there are
# lots of very very small companies out there. 
# 
# In order to limit this to "reasonably sizeable" companies we'll limit our dataset to companies 
# which have at least 100 complaints. This number is rather arbitrary. 
# 
# There are likely some very interesting insights that could be found by looking at "small"
# companies who have quickly accrued complaints, and this data set would be very useful for detecting 
# fraud, but this is outside the scope of the current investigation.

# In[ ]:





# In[ ]:


grouped_by_lcompany = grouped_by_company[grouped_by_company > 100]
print ('Original dataset has {} companies'.format(grouped_by_company.size)) 
print ('Strictly limited dataset has {} companies'.format(grouped_by_lcompany.size)) 


# In[ ]:


company_chart = grouped_by_lcompany.head(25).plot(kind='bar', title='Max Complaints')


# In[ ]:


# This dataframe is then only those companies which have received at least 100 complaints in this
# dataset. From there we can do some deeper digging into those various companies

lcompany_complaints = complaints[complaints.company.isin(grouped_by_lcompany.index)]


# So from here I was going to investigate some interesting information about how each of these 
# companies compared to each other, but the problem is doing this accurately requires one of two 
# things, either information about the size of the company, or information about the products they
# primarily deal in. 
# 
# There's likely some interesting insight into how they respond to complaints, and what their resolution 
# looks like. But for now I'm pausing to tackle this data from a different angle, rather looking
# at what this data looks like by product type.

# In[ ]:




