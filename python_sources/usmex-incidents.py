#!/usr/bin/env python
# coding: utf-8

# I am a beginner, just trying out some pandas and data visualization

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


mm = pd.read_csv('../input/MissingMigrants-Global-2019-03-29T18-36-07.csv')


# In[ ]:


mm.info()


# In[ ]:


mm.head()


# In[ ]:


mm['Region of Incident'].value_counts()


# In[ ]:


#I'll be looking at US-Mexico Border
usmx = mm[mm['Region of Incident'] == 'US-Mexico Border']


# In[ ]:


# date, region and total dead and missing is complete
usmx.info()


# In[ ]:


usmx['Total Dead and Missing'].value_counts()


# In[ ]:


#looking at top 'total dead and missing' entries
usmx.sort_values('Total Dead and Missing',ascending = False).head()


# In[ ]:


# top 15 causes of death
usmx['Cause of Death'].value_counts()[:15].plot(kind = 'barh', figsize = (9,4), color = 'maroon');


# In[ ]:


#total over all 1337 incidents
usmx['Total Dead and Missing'].sum()


# In[ ]:


#filled NaN values with zero
bymn = usmx.groupby(['Reported Year','Reported Month']).agg({'Total Dead and Missing': 'count'}).unstack(fill_value = 0)
bymn


# In[ ]:


#total dead and missing by year and month
bymn.plot(kind = 'bar', legend = False, figsize = (9,4));


# In[ ]:


byyr = usmx.groupby(['Reported Year']).agg({'Total Dead and Missing': 'count'})


# In[ ]:


#total dead and missing by year
byyr.plot(kind = 'bar');

