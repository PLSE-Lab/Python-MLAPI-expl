#!/usr/bin/env python
# coding: utf-8

# **Utility program to check the train.csv daily data file for major errors.
# **
# 
# The John Hopkins data we used in the first week had daily reliability issues, mainly bad country reporting, so it's worth checking the consistency of the data before passing it to the modelling sections.
# 
# These are just some simple checks that flag suspicious entries, the tests thresholds can be modified as needed.
# Needs to be followed up by investigating the suspect dates and specific cleaning.

# In[ ]:


import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')


# In[ ]:


#prevent NA rows dissapear in grouby operations
df_train['Province_State'].fillna(' ',inplace=True)


# In[ ]:


#calculate daily additions to confirmed cases and fatalities in absolute values and percentages
by_ctry_prov = df_train.groupby(['Country_Region','Province_State'])[['ConfirmedCases','Fatalities']]
df_train[['NewCases','NewFatalities']]= by_ctry_prov.transform(lambda x: x.diff().fillna(0))
df_train[['NewCasesPct','NewFatalitiesPct']]= by_ctry_prov.transform(lambda x: x.pct_change().fillna(0))


# In[ ]:


df_train.sort_values('NewCases',ascending = False).head()


# In[ ]:


#check for inconsistencies in daily new cases, cumulative count should only increase or remain equal
df_train[df_train.NewCases < 0].sort_values('NewCases')


# In[ ]:


#check for inconsistencies in daily new fatalities, cumulative count should only increase or remain equal
df_train[df_train.NewFatalities < 0].sort_values('NewFatalities')


# In[ ]:


#more deaths than confirmed cases
df_train[df_train.Fatalities > df_train.ConfirmedCases]


# In[ ]:


#more than 40% increase in ConfirmedCases with at least 1000 new cases - Hubei 13 Feb example
df_train[(df_train.NewCasesPct > 0.4) & (df_train.NewCases > 1000)]


# In[ ]:


#more than 80% increase in ConfirmedCases with at least 50 new cases
df_train[(df_train.NewFatalitiesPct > 0.8) & (df_train.NewFatalities > 50)]


# In[ ]:


#example data cleaning for Hubei/China 13 Feb reporting
#replace day with 14K new cases caused bby measure change in China with average of near dates
maxindx = df_train.loc[(df_train.Country_Region=='China') & (df_train.Province_State=='Hubei'),:].NewCases.idxmax()
df_train.loc[maxindx-2:maxindx+2,:] #before fix of NewCases value for this day


# In[ ]:


avg_smooth = (df_train.NewCases[maxindx-1]+df_train.NewCases[maxindx+1])/2
df_train.loc[maxindx,'NewCases']=avg_smooth
df_train.loc[maxindx-2:maxindx+2,:] #after fix of NewCases value for this day

