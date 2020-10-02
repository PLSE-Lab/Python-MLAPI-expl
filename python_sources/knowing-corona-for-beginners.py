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


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df= pd.read_csv('/kaggle/input/coronavirusdataset/PatientInfo.csv') #_patientinfo
df_cases= pd.read_csv('/kaggle/input/coronavirusdataset/Case.csv') #Data of COVID-19 infection cases in South Korea


# In[ ]:


df.columns


# Exploring Epidemic in Korea- Analyzing what happened at Korea during the Coronavirus Pandemic?!

# In[ ]:


dfK= df.query("country=='Korea'")


# Cases per city

# In[ ]:


casesPercity=dfK.groupby('city')['patient_id'].nunique().sort_values(ascending=False).reset_index().head(10)


# Outbreak of the epidemic in major cities with time

# In[ ]:


dfplot2=dfK[dfK.city.isin(casesPercity['city'].head(10))]
fig, ax= plt.subplots(figsize=(15,7))
dfplot2.groupby(['confirmed_date' ,'city'])['patient_id'].nunique().sort_values(ascending=False).unstack().plot(ax=ax)
# dfK.groupby(['confirmed_date' ,'city_x'])['patient_id_x'].nunique().unstack().plot(ax=ax)
plt.show()


# As it is evident from the above, Gyeongsan-si has the steepest growth in no.of positive cases. Gyeongsan-si lies near the western border abuts the metropolitan city of Daegu (third largest in South Korea after Seoul and Busan) where the outbreak was the maximum.
# 
# **Top cities affected (city/province/infection_reason):**
# 
# 0    Gyeongsan-si/ Gyeongsangbuk-do/ - Daegu has the Shincheonji church!!
# 
# 1      Cheonan-si/Chungcheongnam-do- gym facility in Cheonan
# 
# 2     Seongnam-si

# #### What happened in Daegu? - Shincheonji Church

# In[ ]:


dfplot3=dfK[dfK.infection_case=='Shincheonji Church']
#Distribution of where the 93 people who got affected at Shincheonji Church:
fig, ax= plt.subplots(figsize=(10,7))

dfplot3.groupby(['province'])['patient_id'].nunique().sort_values(ascending= False).plot(kind='bar', ax=ax)
plt.title('Shincheonji Church case- Most people belonged to Daegu')
plt.xlabel('Cities they belonged to')
plt.ylabel('# of people affected')


# In[ ]:


df_cases.query("infection_case=='Shincheonji Church'").sort_values(by='confirmed', ascending=False)


# **Inference**: For the case that occurred in Shincheonji Church, most people arrived from/ went to Gyeongsangnam-do,Gyeongsangbuk-do and Daegu. This is very indicative why the next two places led to major cases of the epidemic.
# 
# Gyeongsangnam-do*(gym facility in Cheonan)*
# 
# Gyeongsangbuk-do*(Guro-gu Call Center)*

# In[ ]:


# Confirmation date vs Age of cases affected at Shincheonji Church:
fig, ax= plt.subplots(figsize=(15,7))
dfplot3.groupby(['confirmed_date','age'])['patient_id'].nunique().unstack().plot(kind= 'bar', stacked=True, ax=ax)
plt.title('Confirmation date vs Age of cases affected at Shincheonji Church')
plt.show()


# Majority of the age group of 60s and 70s were detected the earliest after the incident. 20s and 50s being the major group were detected all along as the symptoms got visible.

# #### Exploring deep into major cases that accelerated the spread of the epidemic

# In[ ]:


fig, ax= plt.subplots(figsize=(10,7))

dfplot3.groupby(['province'])['patient_id'].nunique().sort_values().plot(kind='barh', ax=ax)


# The city's biggest cluster appears to be at a branch of a religious sect which calls itself the Shincheonji Church of Jesus, Temple of the Tabernacle of the Testimony.
# 
# South Korean health officials believe these infections are linked to a 61-year-old woman who tested positive for the virus earlier in Feb. About 1,000 people attended the same service as the woman on Sunday.
# 
# The southern cities of Daegu and Cheongdo have been declared "special care zones". The streets of Daegu are now largely abandoned.
# 
# https://www.bbc.com/news/world-asia-51582186

# #### Overseas inflow

# In[ ]:


dfplot5= dfK[dfK.infection_case=='overseas inflow']
fig, ax= plt.subplots(figsize=(10,7))

dfplot5.groupby(['province'])['patient_id'].nunique().sort_values(ascending= False).plot(kind='bar', ax=ax)
plt.title('Distribution of cases spread via Overseas Inflow- Most people hailed from Seoul/Gyeonggi-do')
plt.xlabel('Cities they belonged to')
plt.ylabel('# of people affected')


# In[ ]:


df_cases.query("infection_case=='overseas inflow'").sort_values(by='confirmed', ascending=False)


# #### Guro-gu Call Center/ Seoul

# In[ ]:


dfplot4= dfK[dfK.infection_case=='Guro-gu Call Center']
fig, ax= plt.subplots(figsize=(10,7))

dfplot3.groupby(['province'])['patient_id'].nunique().sort_values(ascending= False).plot(kind='bar', ax=ax)
plt.title('Distribution of Guro-gu call center case- Most people hailed from Seoul')
plt.xlabel('Cities they belonged to')
plt.ylabel('# of people affected')


# Guro-gu Call Center /Seoul/ cases: 108- As of March 12, 2020 S. Korea confirms 7,755 COVID-19 infections, 99 cases related to cluster spread from Guro-gu call center.

# In[ ]:


df_cases.query("infection_case=='Guro-gu Call Center'").sort_values(by='confirmed', ascending=False)


# #### Onchun Church[](http://)

# In[ ]:


dfplot6= dfK[dfK.infection_case=='Onchun Church']
fig, ax= plt.subplots(figsize=(10,5))

dfplot6.groupby(['province'])['patient_id'].nunique().sort_values(ascending= False).plot(kind='bar', ax=ax)
plt.title('Onchun Church case- Most people hailed from Busan')
plt.xlabel('Cities they hailed from')
plt.ylabel('# of people affected')


# In[ ]:


df_cases.query("infection_case=='Onchun Church'").sort_values(by='confirmed', ascending=False)


# 

# 

# In[ ]:





# 
