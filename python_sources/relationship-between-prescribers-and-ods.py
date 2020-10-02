#!/usr/bin/env python
# coding: utf-8

# 

# Start by importing standard libraries and data

# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import re


# In[ ]:


prescribers = pd.read_csv('../input/prescriber-info.csv')
opioids = pd.read_csv('../input/opioids.csv')
ODs = pd.read_csv('../input/overdoses.csv')


# In[ ]:


import re
ops = list(re.sub(r'[-\s]','.',x) for x in opioids.values[:,0])
prescribed_ops = list(set(ops) & set(prescribers.columns))


# In[ ]:


prescribers['NumOpioids'] = prescribers.apply(lambda x: sum(x[prescribed_ops]),axis=1)
prescribers['NumPrescriptions'] = prescribers.apply(lambda x: sum(x.iloc[5:255]),axis=1)
prescribers['FracOp'] = prescribers.apply(lambda x: float(x['NumOpioids'])/x['NumPrescriptions'],axis=1)


# In[ ]:


prescribers.plot.scatter('NumOpioids','NumPrescriptions')


# In[ ]:


prescribers.hist('FracOp')


# In[ ]:


mean_NO = prescribers.groupby('Specialty')['NumOpioids'].mean().sort_values(ascending=False)
mean_fracO = prescribers.groupby('Specialty')['FracOp'].mean().sort_values(ascending=False)
mean_NO.head()


# In[ ]:


prescribers['O.Diff'] = prescribers.apply(lambda x: x['NumOpioids'] - mean_NO[x['Specialty']],axis=1)
prescribers['FracO.Diff'] = prescribers.apply(lambda x: x['FracOp'] - mean_fracO[x['Specialty']],axis=1)


# In[ ]:


prescribers


# Next, we group the prescribers by state, ignoring any State values that are not in our OD map (which only contains the 50 states). Then we create Series of the total number of prescribers per state and the number of those who are frequent opioid prescribers

# In[ ]:


p = prescribers[prescribers['State'].isin(ODs['Abbrev'])]
op = p[p['Opioid.Prescriber']==1]['State'].value_counts()
tot_p = p['State'].value_counts()


# Next, we clean up the OD data, converting to floats

# In[ ]:


ODs['Deaths'] = ODs['Deaths'].apply(lambda x: x.replace(',','')).astype('float')
ODs['Population'] = ODs['Population'].apply(lambda x: x.replace(',','')).astype('float')


# Now, we calculate the number of OD deaths per million population and add the prescriber data to the OD dataframe

# In[ ]:


ODs['DeathsPerCap'] = ODs['Deaths']/ODs['Population']*1E6
ODs['TotalPrescribers'] = ODs['Abbrev'].apply(lambda x: tot_p[x])
ODs['OPrescribers'] = ODs['Abbrev'].apply(lambda x: op[x])
ODs['FracOPrescribers'] = ODs['Abbrev'].apply(lambda x: op[x]/tot_p[x])


# Finally, we look at the correlation between the fraction of opiod prescribers and OD deaths per capita. There does not seem to be a strong relationship.

# In[ ]:


ODs.plot.scatter('FracOPrescribers','DeathsPerCap')


# In[ ]:


print("Correlation: %4f" % (ODs['FracOPrescribers'].corr(ODs['DeathsPerCap'])))


# In[ ]:




