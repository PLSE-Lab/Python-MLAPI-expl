#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##LOADING csv file

import os, sys, email
import numpy as np 
import pandas as pd
# Plotting
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set_style('whitegrid')
#import plotly
#plotly.offline.init_notebook_mode()
#import plotly.graph_objs as go
import wordcloud

# Network analysis
import networkx as nx
# NLP
from nltk.tokenize.regexp import RegexpTokenizer

from subprocess import check_output
print(check_output(["ls", "../input/enron-email-dataset/"]).decode("utf8"))


# In[ ]:


#Read data into DataFrame
df = pd.read_csv('../input/enron-email-dataset/emails.csv')
print(df.shape)
df.head()


# In[ ]:


#Sample Email from the dataset
print(df['message'][3])


# In[ ]:


##CLEANING data

#Create Helper functions (From Zichen Wang's Kernel: Explore Enron)
def get_text_from_email(msg):
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain': #extracting text data
            parts.append(part.get_payload())
    return ''.join(parts)

def split_email_adds(line): #splitting email addresses
    if line:
        addrs = line.split(',')
        addrs = frozenset(map(lambda x : x.strip(), addrs))
    else:
        addrs = None
    return addrs


# In[ ]:


#create list of email objects
msgs = list(map(email.message_from_string, df['message']))
df.drop('message', axis=1, inplace=True) # axis = 1 used to apply a method across each row
# Get all fields from the email objects
fields = msgs[0].keys()
for field in fields:
    df[field] = [doc[field] for doc in msgs]
    
#parse content from emails
df['content'] = list(map(get_text_from_email, msgs))
# Split email address
df['From'] = df['From'].map(split_email_adds)
df['To'] = df['To'].map(split_email_adds)

#Extract the root of file as user
df['user'] = df['file'].map(lambda x:x.split('/')[0])
del msgs

df.head()


# In[ ]:


print(df.shape)
for col in df.columns:
    print(col, df[col].nunique())


# In[ ]:


df = df.set_index('Message-ID').drop(['file','Mime-Version','Content-Type', 'Content-Transfer-Encoding'], axis=1)
df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format = True)
df.dtypes


# In[ ]:


#Creating from_user list
userList = []
userList = df['user'].unique()


# In[ ]:


#Number of emails sent by each users

count = []
counter = 0 
for l in userList:
    for user in df['user']:
        if user == l:
            counter += 1
    count.append(counter)
    counter = 0


# In[ ]:


'''Let us take a look at how many email were sent by each user. 
This plot may indicate erratic email usage. 
We plot top 20 users who sent the highest number of emails.'''

new_list = sorted(list(zip(count, userList)), reverse = True)[:20] ##merge both lists and sort top 20, 
print(new_list)
num, mailer = zip(*new_list) ## separate lists previously merged and use them for plotting
mail = range(len(new_list))
plt.bar(mail, num, align = 'center', color ='green', alpha=0.8)
plt.xticks(mail, mailer, rotation='vertical')
plt.show()

'''The histogram shows users vs no. of emails they sent'''


# In[ ]:


#Read data into DataFrame
newdf = pd.read_csv('../input/year-vs-number-of-emails-enron-emails/part.csv')
print(newdf.shape)
newdf.head()


# In[ ]:


#The dataset imported in the previous lines was extracted using MapReduce on CloudEra Hadoop Distribution 
#from the enron-email dataset. The new dataset retrieved as a result was uploaded to kaggle in the name of
#"year-vs-number-of-emails-enron-emails".
#The dataset includes three columns -
#User - name of the user
#Year - Year in which the user sent emails
#Emails - Number of emails sent by a particular user in a particular year.
#Using the data above, we will plot a graph to look for a user who sent the most emails around the time of the scam.'''
user = newdf["user"]
year = newdf[" year"]
emails = newdf[" emails"]
listnew = sorted(list(zip(emails, user, year)), reverse = True)[:21] ##merge both lists and sort top 15,
dframe = pd.DataFrame(listnew)
dframe.columns = ['emails', 'user', 'year']
dframe.head()
#The list 'listnew' contains top users with most emails sent"

#plot
fig, ax = plt.subplots()
fig.set_size_inches(10, 5)
sns.barplot(x='user', y='emails', hue='year', data=dframe, saturation=0.5)
sns.despine()
plt.xticks(rotation=45)
plt.legend(loc='upper right')
fig.savefig('example.png')
plt.xlabel('Users')
plt.ylabel('Number of emails')


'''The graph shows which user sent the most emails in which year.'''


# In[ ]:


# The number of emails indicates a pattern,however it also important to consider the network
# that existed in the company.
dfnew = df.head(n=350)

toUSER = []
for user in dfnew['X-To']:
    user = user.split('<')[0]
    user = user.split('@')[0]
    toUSER.append(user)
newlister = list(zip(dfnew['user'],toUSER))
datanewf = pd.DataFrame(newlister)
datanewf.columns = ['fromuser', 'touser']
G = nx.from_pandas_dataframe(datanewf, 'fromuser', 'touser')
plt.figure(figsize=(15,15))
pos = nx.draw_random(G, node_size = 25, node_color = 'red', edge_color = 'brown', with_labels = True)
plt.title('Network of emails (First 350)')
plt.show()

'''In this graph, I attempted to visualize a network with users who sent and received emails as nodes. 
I chose to display only the first 350 emails because the graph looks clumsy with labels. 
From this graph, it can be noticed that the point of centrality is allen-p.'''


# In[ ]:


dfnew = df.head(n=1000)

toUSER = []
for user in dfnew['X-To']:
    user = user.split('<')[0]
    user = user.split('@')[0]
    toUSER.append(user)
newlister = list(zip(dfnew['user'],toUSER))
datanewf = pd.DataFrame(newlister)
datanewf.columns = ['fromuser', 'touser']
G = nx.from_pandas_dataframe(datanewf, 'fromuser', 'touser')
plt.figure(figsize=(30,30))
pos = nx.draw_circular(G, node_size = 50, node_color = 'black', edge_color = 'black', with_labels = False)
plt.show()

'''The graph displays the network of first 1000 emails. The point of centrality is at node representing allen-p. 
Therefore, in this set of 1000 emails, the allen-p seems influential. However, this analysis is inaccurate as 
we did not consider the total of 517000 emails.'''


# In[ ]:




