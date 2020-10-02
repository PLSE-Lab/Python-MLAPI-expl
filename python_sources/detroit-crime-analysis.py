#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # SAMPLE DATA FOR CRIME IN DETROIT

# In[ ]:


df=pd.read_csv("../input/DPD__All_Crime_Incidents__January_1__2009_-_December_6__2016.csv")
df.head()


# # CRIMES vs FREQUENCY

# In[ ]:


plt.figure(figsize=(15,15))
df.CATEGORY.value_counts().plot('bar')
plt.xlabel('Type of crime')
plt.ylabel('frequency')


# # NEIGHBOUHOOD-WISE CRIMES vs FREQUENCY

# In[ ]:


for i in df.COUNCIL.unique():
    if pd.isnull(i):
        continue
    plt.figure()
    df[df['COUNCIL']==i].NEIGHBORHOOD.value_counts().plot('bar')
    plt.title(i)
    plt.xlabel('Neighbourhood where crimes took place')
    plt.ylabel('frequency')


# # DISTRICT-WISE CRIMES vs FREQUENCY

# In[ ]:


df.COUNCIL.value_counts().plot('bar')
plt.xlabel('Crime in District')
plt.ylabel('Frequency')


# # DISTRICT-WISE DISTRIBUTION OF DATA WITH FREQUENCY OF EACH CRIME

# In[ ]:


for i in list(df.COUNCIL.unique()):
    if pd.isnull(i):
        continue
    fig=plt.figure()
    df[df.COUNCIL==i].CATEGORY.value_counts().plot.bar()
    plt.title(i)
    plt.xlabel('Crimes')
    plt.ylabel('Frequency')
    


# # CRIME-WISE DISTRIBUTION OF DATA WITH FREQUENCY OF THAT CRIME IN EACH COUNCIL

# In[ ]:


#print(type(df[df.CATEGORY==i].COUNCIL.value_counts()))
for i in list(df.CATEGORY.unique()):
    if i=='SEX OFFENSES':
        continue
    fig=plt.figure()
    df[df.CATEGORY==i].COUNCIL.value_counts().plot.bar()
    plt.title(i)
    plt.xlabel('COUNCIL')
    plt.ylabel('Frequency')
    


# In[ ]:


import datetime


# # HOUR-WISE DISTRIBUTION OF DATA WITH FREQUENCY OF EACH CRIME 

# In[ ]:


for i in df.HOUR.sort_values().unique():
    fig=plt.figure()
    df[df.HOUR==i].CATEGORY.value_counts().plot.bar()
    plt.title(('{} HOUR').format(i))
    plt.xlabel('CATEGORY')
    plt.ylabel('Frequency')


# # CRIME-WISE DISTRIBUTION OF DATA WITH FREQUENCY OF CRIMES IN DIFFERENT HOURS

# In[ ]:


for i in df.CATEGORY.unique():
    fig=plt.figure()
    tmp=df[df.CATEGORY==i].HOUR.value_counts()#.plot.bar()
    tmp.sort_index().plot.bar()
    plt.xlabel('HOUR')
    plt.title(i)
    plt.ylabel('Frequency')


# In[ ]:


#df['DATE'],df['Time']=
#x=df['INCIDENTDATE'].str.split(" ",expand=True)
#df['DATE']=
len(df['INCIDENTDATE'])
df['DATE']=pd.to_datetime(df['INCIDENTDATE'])
#df.DATE.value_counts().plot()


# In[ ]:


plt.figure(figsize=(20,20))
df.DATE.value_counts().plot()


# In[ ]:





# In[ ]:




