#!/usr/bin/env python
# coding: utf-8

# I referenced others' notebooks and tried to get familiar with Python since I'm new here.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Let's get the dataset loaded.

# In[ ]:


df = pd.read_csv("../input/Tweets.csv")


# Do some EDAs.

# In[ ]:


df.shape


# In[ ]:


Mood = df['airline_sentiment'].value_counts()
Mood


# In[ ]:


index = [1,2,3]
plt.bar(index,Mood,color=['r','b','g'])
plt.xticks(index,['Negative','Neutral','Positive'])
plt.xlabel('Mood')
plt.ylabel('Mood Count')
plt.title('Mood Distribution')


# In[ ]:


Airline_count = df['airline'].sort_index().value_counts()
Airline_count.plot(kind='bar',rot=45)
plt.show()


# In[ ]:


def plot_sub_sentiment(Airline):
    pdf = df[df['airline']==Airline]
    count = pdf['airline_sentiment'].value_counts()
    Index = [1,2,3]
    color = ['red','blue','green']
    plt.bar(Index,count,width=0.5,color=color)
    plt.xticks(Index,['Negative','Neutral','Positive'])
    plt.title('Mood Summary of' + " " + Airline)

airline_name = df['airline'].unique()
plt.figure(1,figsize=(12,12))
for i in range(6):
    plt.subplot(3,2,i+1)
    plot_sub_sentiment(airline_name[i])
plt.show()


# In[ ]:


NR_Count = dict(df['negativereason'].value_counts(sort=False))
def NR_Count(Airline):
    if Airline=='All':
        NR_df=df
    else:
        NR_df=df[df['airline']==Airline]
    count=dict(NR_df['negativereason'].value_counts())
    Unique_reason=list(df['negativereason'].unique())
    Unique_reason=[x for x in Unique_reason if str(x) != 'nan']
    Reason_frame=pd.DataFrame({'Reasons':Unique_reason})
    Reason_frame['count']=Reason_frame['Reasons'].apply(lambda x: count[x])
    return Reason_frame

def plot_reason(Airline):
    NR_df=NR_Count(Airline)
    count=NR_df['count']
    Index = range(1,(len(NR_df)+1))
    plt.bar(Index,count)
    plt.xticks(Index,NR_df['Reasons'],rotation=90)
    plt.ylabel('Count')
    plt.xlabel('Reason')
    plt.title('Count of Reasons for '+Airline)


# In[ ]:


plot_reason('All')
plt.show()


# In[ ]:


airline_names = df['airline'].unique()
plt.figure(1,figsize=(12,36))
for i in range(0,6):
    plt.subplot(3,2,i+1)
    plot_reason(airline_names[i])
    plt.xlabel('')
    plt.ylim(0,800)
plt.show()


# TBD
