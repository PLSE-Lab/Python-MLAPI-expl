#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pylab
from datetime import datetime
import seaborn as sns
import re
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")

turnos = pd.read_csv('../input/KaggleV2-May-2016.csv')


# In[ ]:


turnos.head()


# In[ ]:


turnos.info()


# In[ ]:


turnos.ScheduledDay = turnos.ScheduledDay.apply(np.datetime64)
turnos.AppointmentDay = turnos.AppointmentDay.apply(np.datetime64)
turnos['WaitTime'] = (turnos.AppointmentDay - turnos.ScheduledDay).dt.days
turnos['No-show']= turnos['No-show'].apply(lambda x: 0 if x =="No" else 1)
turnos['GenderB']= turnos['Gender'].apply(lambda x: 0 if x =="M" else 1)
turnos['weekday'] = turnos.AppointmentDay.dt.weekday


# In[ ]:


turnos.describe()


# In[ ]:


turnos[turnos['Handcap'] > 0]['Handcap'].value_counts()


# In[ ]:


turnos[turnos['WaitTime'] < 0]['WaitTime'].value_counts()


# In[ ]:


def ageRange (x):
    if x < 0 : return '0-inUtero'
    elif x >=0 and x < 6 : return '1-PInfancia'
    elif x >=6 and x < 12 : return '2-Infancia'
    elif x >=12 and x < 14 : return '3-Adolescencia'
    elif x >=14 and x < 27 : return '4-Juventud'
    elif x >=27 and x < 60 : return '5-Adultez'
    else: return '6-Vejez'

turnos['AgeClass']= turnos['Age'].apply(ageRange)


# In[ ]:


turnos['Handcap'] = turnos['Handcap'].apply(lambda x: 1 if x != 0 else x)


# In[ ]:


for i, turno in turnos.iterrows():
    #turnos.loc[i, "Recurrent"] = turnos[(turnos.PatientId ==turno.PatientId) & (turnos.AppointmentDay <= turno.ScheduledDay) & (turnos['No-show'] == 1)].count()
    turnos.loc[i, "Recurrent"]= turnos[(turnos.PatientId ==turno.PatientId) & (turnos.AppointmentDay <= turno.ScheduledDay) & (turnos['No-show'] == 1)]['No-show'].count()


# In[ ]:


target = sns.countplot(x="No-show", data=turnos)


# In[ ]:


turnos = turnos[turnos.Age > 0]
turnos = turnos[turnos.Age <= 100]
turnos = turnos[turnos.WaitTime >= -1]


# In[ ]:


turnos.groupby('No-show').Age.plot(kind='kde')


# In[ ]:


age_hist = sns.boxplot(x="No-show", y="Age", data=turnos)


# In[ ]:


sns.stripplot(turnos.WaitTime, jitter=True)


# In[ ]:


sns.countplot(x="AgeClass", hue = "No-show", data=turnos)

