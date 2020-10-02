#!/usr/bin/env python
# coding: utf-8

# <h2>Importing the needed libraries </h2>

# In[ ]:


import numpy as np ##for linear algebra
import pandas as pd #data cleaning and wrangling 
import re ##to extract strings using a parttern 
import emoji #to acess emoji
import matplotlib.pyplot as plt #visualisation
import seaborn as sns #visualisation


# <h2>Importing the Data</h2>

# In[ ]:


d=open(r"C:\Users\User\Downloads\WhatsApp Chat with Discussing Football",mode='r', encoding = 'utf-8')
data=d.read()


# <h2>Checking to see the Data</h2>

# In[ ]:


data


# # Extracting the date from the data

# In[ ]:


datep=re.compile('\d+/\d+/\d+')
date=re.findall(datep, data)
date


# # Extracting time from the bulk of text

# In[ ]:


timep=re.compile('\d+:\d+')
time=re.findall(timep, data)
time


# # Extracting the names of message senders 

# In[ ]:


senderp=re.compile('-.*?:')
sender1=re.findall(senderp, data)
sender1


# # Getting the messages 

# In[ ]:


messdata=re.sub('\d+/\d+/\d+,\s\d+:\d+\s-','',data)


# In[ ]:


messdata


# # Alternatively...
# You could split each line and remove the date, time, names and messages into lists 

# In[ ]:


messagep=re.compile('\d+/\d+/\d+,\s\d+:\d+\s-.*:.*')
dtu=re.findall(messagep, data)
dtu


# In[ ]:


date=list(map(lambda t : t.split("-")[0],dtu))
#split the lines into 2 using the '-' sign


# In[ ]:


len(date)


# In[ ]:


date


# In[ ]:


time=list(map(lambda x:x.split(',')[1].strip(' '),date))
time
#Seperate the time with the comma and put it in a list 


# In[ ]:


len(time)


# In[ ]:


date=list(map(lambda x:x.split(',')[0].strip(' '),date))
date


# In[ ]:


sender_message=list(map(lambda t : t.split("-")[1],dtu))
sender_message
#getting the names and the messengers 


# In[ ]:


sender=list(map(lambda sm: sm.split(':')[0].lstrip(' '), sender_message))
sender


# In[ ]:


message=list(map(lambda sm: sm.split(':')[1].lstrip(' '), sender_message))
message
#getting the messages 


# In[ ]:


df=pd.DataFrame({'Date':date, 'Time':time, 'Sender':sender, 'Message':message})
#creating a dataframe from what was collected 


# In[ ]:


df.index+=1
df


# In[ ]:


df["Letter_Count"]=df['Message'].apply(lambda w: len(w.replace(' ','')))


# In[ ]:


df["Word_Count"]=df['Message'].apply(lambda w: len(w.split(' ')))


# In[ ]:


df['Avg_Word_length']=df['Letter_Count']//df["Word_Count"]


# In[ ]:


df.to_csv(r"C:\Users\User\Downloads\Discussing Football.csv", encoding='utf-8-sig')


# In[ ]:


df[df['Word_Count']==df['Word_Count'].max()]


# In[ ]:


longest_sentences=df.sort_values(by=['Word_Count'],ascending=False).head(5)


# In[ ]:


df.sort_values(by=['Letter_Count'],ascending=False).head(5)


# In[ ]:


df.sort_values(by=['Avg_Word_length'],ascending=False).head(5)


# In[ ]:


number_of_messages=df.Sender.value_counts()


# In[ ]:


Active_members=number_of_messages.head(5)


# In[ ]:


least_activeMembers=number_of_messages.tail(5)


# In[ ]:


df.query('Message=="You deleted this message"')


# In[ ]:


df=df.drop([5078,19199,19200], axis=0)


# In[ ]:


todrop=df.query('Message=="This message was deleted"')


# In[ ]:


df['Message'].value_counts()["This message was deleted"]


# In[ ]:


todrop["Sender"].value_counts().head(5)


# In[ ]:


df['Hour']=df['Time'].apply(lambda t:t.split(':')[0]).astype('int64')
df["Hour"]


# In[ ]:


df['Hour'].value_counts().head(10)


# In[ ]:


df['Hour'].value_counts().tail(5)


# In[ ]:


df['Hour'].dtypes


# In[ ]:


nocturnals=df[df['Hour']<6]


# In[ ]:


nocturn=nocturnals['Sender'].value_counts()


# In[ ]:


df.to_csv(r"C:\Users\User\Downloads\Discussing Football.csv", encoding='utf-8-sig',index=False)


# In[ ]:



plt.axes([1,1,1,0.98])
plt.grid(True)
nocturn.plot.bar()
plt.xlabel('Guys That Message at Night')
plt.ylabel('No. of Messages')
plt.xticks(rotation=90)
plt.show()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


rush_hours=df['Hour'].value_counts().head(10)
rush_hours.sort_index().plot.bar()
plt.xlabel('hours')
plt.ylabel('No. of Messages')
plt.xticks(rotation=0)
plt.show()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


rush_hours.sort_index()


# In[ ]:



df['Hour'].value_counts().tail(10).sort_index().plot.bar()
plt.xlabel('hours')
plt.ylabel('No. of Messages')
plt.xticks(rotation=0)
plt.show()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df.groupby(['Date']).mean().sort_values(by='Date')


# In[ ]:


no_msg_pday=df.groupby(['Date']).count()


# In[ ]:


no_msg_pday['Message'].mean()


# In[ ]:


df['Message'].count()


# In[ ]:


df.to_csv(r"C:\Users\User\Downloads\Discussing Football.csv", encoding='utf8-sig')


# In[ ]:




