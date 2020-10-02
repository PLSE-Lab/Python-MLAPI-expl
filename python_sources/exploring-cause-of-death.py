#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Read in the death records to pandas df
deaths = pd.read_csv('../input/DeathRecords.csv')
codes = pd.read_csv('../input/Icd10Code.csv')


# ##Deaths from alcohol vs. narcotics
# First up, we look at the number of deaths from alcohol vs. those from narcotics (or at least deaths with those terms listed in the official ICD10 Code)
# 

# In[ ]:


alc = list(codes[codes['Description'].str.contains('alcohol')]['Code'])
narc = list(codes[codes['Description'].str.contains('narcotics')]['Code'])

print('Alcohol:', deaths[deaths['Icd10Code'].isin(alc)].shape[0])
print('Narcotics:', deaths[deaths['Icd10Code'].isin(narc)].shape[0])


# #Suicides by age
# A histogram of suicides by age. There is a large peak in the middle of the distribution around 50 years of age, I
# suspect this is a result of the overall age distribution in the US, but would need more data to confirm
# 

# In[ ]:


deaths[deaths['MannerOfDeath']==2]['Age'].hist(bins=range(102))


# #Law Enforcement Deaths vs. amount of education
# ICD10 codes beginning with Y35 are deaths resulting from interaction with law enforcement. Here I compare deaths resulting form law enforcement intervention with the deceased's level of education
# 

# In[ ]:


d = pd.merge(deaths,codes,left_on='Icd10Code',right_on='Code')
ed = pd.read_csv('../input/Education2003Revision.csv')
print(ed['Description'])
d[d['Icd10Code'].str.contains('Y35')]['Education2003Revision'].hist(bins=range(8))


# #Causes more deadly than terrorists
# A list of causes of death that claimed as many or more lives than terrorist attacks did in 2014. I limited this list to deaths with an ICD10 Code beginning with R-Z, as those are the codes associated with abnormal or external causes of death. This was done to focus on deaths not directly caused by health or disease problems. 
# 
# According to http://securitydata.newamerica.net/extremists/deadly-attacks.html, there were 12 deaths due to terrorist attacks in 2014
# 

# In[ ]:


count = d[d['Icd10Code']>='R']['Description'].value_counts()
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_colwidth',100)
count[count.values>=12].sort_values(ascending=True).to_frame()

