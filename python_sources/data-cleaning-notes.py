#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
payroll = pd.read_csv("../input/Citywide_Payroll_Data__Fiscal_Year_.csv")


# In[ ]:


payroll.head()


# As with much city data, this data comes in with a few schematic kinks in it that need to be massaged out. For example, see here how POLICE DEPARTMENT appears in two different ways in the rolls:

# In[ ]:


payroll['Agency Name'].value_counts()[['POLICE DEPARTMENT', 'Police Department']]


# To correct this here and elsewhere let's uppercase the offending columns.

# In[ ]:


payroll['Agency Name'] = payroll['Agency Name'].str.upper()
payroll['Work Location Borough'] = payroll['Work Location Borough'].str.upper()


# Note that the *Borough* in `Work Location Borough` is kind of a misnomer, as NYC records some people working for it as far off as Washington DC.

# In[ ]:


payroll['Work Location Borough'].value_counts()[::-1].head()


# Next, we need to strip whitespace out of the title descriptions. For example

# In[ ]:


payroll['Title Description'].value_counts().index[11]


# In[ ]:


payroll['Title Description'] = payroll['Title Description'].str.strip()


# Another thing to be aware of when you're working on this dataset is that certain departments are split up in a way you may or may not agree with or find useful. For example, there are six categories for New York City Department of Education staff (which is, not surprisingly, the largest city government division):

# In[ ]:


[(d, count) for (d, count) in payroll['Agency Name'].value_counts().iteritems()    if ' ED ' in d or 'EDUCATION' in d]


# Additionally, note that this dataset includes everyone that has ever received compensation from the City of New York within the given time period. That means that it includes temporary worker, folks with titles like ELECTION WORKER (who come onto the payroll during election season) and JOB TRAINING PARTICIPANT (certain job training programs run by the City pay impoverished public assistance users to work part time and learn jobholding skills part time).

# In[ ]:


len(payroll[payroll['Title Description'] == 'JOB TRAINING PARTICIPANT'])

