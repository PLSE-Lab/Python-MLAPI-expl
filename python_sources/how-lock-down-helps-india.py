#!/usr/bin/env python
# coding: utf-8

# #                                                      COVID-19 in India
# Corona Virus which orginated from China in Wuhan has now spreaded to over all countries in the world. Due to corona virus many people lost their lives.European countries are heavily affected by this virus. More than 500 people die in countries like Spain,Italy and USA.Virus also attacked many asian countries.Currently Iran is most affected to this virus next to China in Asia.However China is now recovering from Virus spread. It has taken many actions to control the spread of virus. Now the number of infected person have became very low in China. In India the precautionary measures like lock down were taken very soon. By this a large amount of infections can be brought into control. Here we will see what will be the infection rate if lock down was not implimented till April 13.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[ ]:


data=pd.read_csv('../input/covid19-corona-virus-india-dataset/complete.csv')


# In[ ]:


data.head()


# In[ ]:


data['Date']=pd.to_datetime(data['Date'])


# In[ ]:


data.set_index("Date",inplace=True)


# In[ ]:


data.head()


# In[ ]:


data['Name of State / UT'].unique()


# In[ ]:


data.describe()


# First we will see the effects that will take place in Tamil Nadu. 

# In[ ]:


TN=data[data['Name of State / UT']=='Tamil Nadu']


# In[ ]:


TN.head()


# In[ ]:


TN['Today_case']=0
for i in TN.index:
    a=i-pd.DateOffset(days=1)
    if a not in TN.index:
        a=i
    TN['Today_case'].loc[i]=(TN['Total Confirmed cases'].loc[i])-(TN['Total Confirmed cases'].loc[a])


# In[ ]:


TN.tail(10)


# For prediction it is necessary to find the transmission rate in the state. This can be achieved by dividing the number of cases that came today to that of the number of cases that came yesterday. With the total set of transmission rate we can then find the mean of it. The mean value gives transmission rate. It is necessary to find the date from where the transmission started. By doing so we could get results with better accuracy.

# In[ ]:


import math
TN['Transmission_rate']=0
for i in TN.index:
    a=i-pd.DateOffset(days=1)
    if a not in TN.index:
        a=i
    TN['Transmission_rate'].loc[i]=(TN['Today_case'].loc[i])/(TN['Today_case'].loc[a])
    if TN['Transmission_rate'].loc[i]==math.inf:
        TN['Transmission_rate'].loc[i]=0


# In[ ]:


TN.tail()


# In[ ]:


TN['Transmission_rate'].fillna(value=0)


# In[ ]:


trans_start=TN['Transmission_rate'].loc['2020-03-23':]
trans_start


# In[ ]:


overall_trans=trans_start.mean()
overall_trans


# After finding the transmission rate we can build a model with it which gives the value of number of persons infected when lockdown is not implemented.

# In[ ]:


TN['No_lock_affected']=0
TN['No_lock_affected'].loc['2020-03-23']=TN['Total Confirmed cases'].loc['2020-03-23']
for i in TN.index:
    if i in pd.date_range('2020-03-24',freq='D',periods=30):
        a=i-pd.DateOffset(days=1)
        if a not in TN.index:
            a=i
        TN['No_lock_affected'].loc[i]=TN['No_lock_affected'].loc[a]*overall_trans


# In[ ]:


TN.head()


# In[ ]:


fig=plt.figure(figsize=(20,5))
plt.plot(TN.index,TN['Total Confirmed cases'],'g',label='Actual Case')
plt.plot(TN.index,TN['No_lock_affected'],'r',label='Cases without lockdown')
plt.title('Tamil Nadu COVID-19 Cases')
plt.legend()


# From the above graph it is evident that if the lock down was not implemented correctly the no of cases in Tamil Nadu goes to above 180000. By this graph we could find the pros of lock down at this pandemic situation. 

# Further we are going to predict for 10 most corona affected states in India. For that we are going to reduce the large number of codes by using functions.

# In[ ]:


def data_inclusion(data):
    data['Today_case']=0
    for i in data.index:
        a=i-pd.DateOffset(days=1)
        if a not in data.index:
            a=i
        data['Today_case'].loc[i]=(data['Total Confirmed cases'].loc[i])-(data['Total Confirmed cases'].loc[a])
        
    data['Transmission_rate']=0
    for i in data.index:
        a=i-pd.DateOffset(days=1)
        if a not in data.index:
            a=i
        data['Transmission_rate'].loc[i]=data['Today_case'].loc[i]/data['Today_case'].loc[a]
        if data['Transmission_rate'].loc[i]==math.inf:
            data['Transmission_rate'].loc[i]=0
    data['Transmission_rate'].fillna(value=0,inplace=True)
    return data


# In[ ]:


def without_lock(data):
    trans_start=data['Transmission_rate'].loc['2020-03-30':]
    overall_trans=trans_start.mean()
    if overall_trans<1.0:
        overall_trans=overall_trans+0.4
    
    data['No_lock_affected']=0
    data['No_lock_affected'].loc['2020-03-23']=data['Total Confirmed cases'].loc['2020-03-23']
    for i in data.index:
        if i in pd.date_range('2020-03-24',freq='D',periods=30):
            a=i-pd.DateOffset(days=1)
            if a not in data.index:
                a=i
            data['No_lock_affected'].loc[i]=data['No_lock_affected'].loc[a]*overall_trans
    return data,overall_trans


# In[ ]:


def graph(data):
    fig=plt.figure(figsize=(20,5))
    plt.plot(data.index,data['Total Confirmed cases'],'g',label='Actual Case'),
    plt.plot(data.index,data['No_lock_affected'],'r',label='Cases without lockdown'),
    plt.title(data['Name of State / UT'].unique() +' COVID-19 Cases')
    plt.legend()


# In[ ]:


def prediction(data):
    data_inclusion(data)
    without_lock(data)
    graph(data)


# #                                    Covid Cases in Maharashtra

# In[ ]:


MH=data[data['Name of State / UT']=='Maharashtra']
MH.head()


# In[ ]:


prediction(MH)


# #                                       Covid Cases in Delhi

# In[ ]:


DL=data[data['Name of State / UT']=='Delhi']
DL.head()


# In[ ]:


prediction(DL)


# #                                     Covid Cases in Rajasthan

# In[ ]:


RJ=data[data['Name of State / UT']=='Rajasthan']
RJ.head()


# In[ ]:


prediction(RJ)


# #                                        Covid Cases in Telengana

# In[ ]:


TL=data[data['Name of State / UT']=='Telengana']
TL.head()


# In[ ]:


prediction(TL)


# #                                    Covid Cases in Madhya Pradesh

# In[ ]:


MP=data[data['Name of State / UT']=='Madhya Pradesh']
MP.head()


# In[ ]:


prediction(MP)


# #                                          Covid Cases in Gujarat

# In[ ]:


GJ=data[data['Name of State / UT']=='Gujarat']
GJ.head()


# In[ ]:


prediction(GJ)


# #                                      Covid Cases in Uttar Pradesh

# In[ ]:


UP=data[data['Name of State / UT']=='Uttar Pradesh']
UP.head()


# In[ ]:


prediction(UP)


# #                                        Covid Cases in Kerala

# In[ ]:


KL=data[data['Name of State / UT']=='Kerala']
KL.head()


# In[ ]:


prediction(KL)


# #                                        Covid Cases in Karnataka

# In[ ]:


KA=data[data['Name of State / UT']=='Karnataka']
KA.head()


# In[ ]:


prediction(KA)


# These graphs explain us why we need lock down. It is better to stay isolated and maintain social distancing inorder to keep ourselves healthy. So always maintain social distancing and be safe. Thank You
