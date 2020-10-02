#!/usr/bin/env python
# coding: utf-8

# # Sample Submission for January's Featured Dataset Task
# 
# * Jan: Human Trafficking Prevention Month
#  * Create and describe a plot that relates to the theme "Jan: Human Trafficking Prevention Month"

# Human Trafficking is a global problem.  The National Referral Mechanism contains records of human trafficking victims in the United Kingdom.  The number of reported incidents from 1745 in 2013 to 3266 in 2015.  Likewise, the number of positive conclusions has increased from 824 incidents in 2013 to 1028 incidents in 2015.  This is a tragedy.  It is important that additional resources are allocated to help prevent human trafficking in future years.

# In[ ]:


import pandas as pd
decision_data = pd.read_csv('/kaggle/input/uk-human-trafficking-data/2016_decision_data.csv')
decision_data[:-1].plot.line(x='Year', 
                             y=['Total Number of Referrals','Positive Conclusive Decisions'],
                             figsize=(10,10))


# In[ ]:




