#!/usr/bin/env python
# coding: utf-8

# # COVID-19 containment and mitigation measures
# 
# Using data from http://epidemicforecasting.org/containment

# In[ ]:


import pandas as pd
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_colwidth', 150)
df = pd.read_csv('/kaggle/input/covid19-containment-and-mitigation-measures/COVID 19 Containment measures data.csv')
df = df[['Country','Date Start','Description of measure implemented']].sort_values('Date Start',ascending=False)
df.to_csv('/kaggle/working/containment_and_mitigation_measure.csv')
df.head(1703)


# In[ ]:


print('# of entries: = ',df.shape[0])


# In[ ]:




