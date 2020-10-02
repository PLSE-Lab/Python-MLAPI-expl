#!/usr/bin/env python
# coding: utf-8

# # Alice vs. Intruder: Simple feature search and exploratory data analysis #

# ** Import libraries and set desired options **

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')


# **Read training and test sets, sort train set by session start time.**

# In[5]:


train_df = pd.read_csv('../input/train_sessions.csv',
                       index_col='session_id')
test_df = pd.read_csv('../input/test_sessions.csv',
                      index_col='session_id')

# Convert time1, ..., time10 columns to datetime type
times = ['time%s' % i for i in range(1, 11)]
train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

# Sort the data by time
train_df = train_df.sort_values(by='time1')

# Look at the first rows of the training set
train_df.head()


# **Transform data into format which can be fed into `CountVectorizer` **

# In[6]:


sites = ['site%s' % i for i in range(1, 11)]
train_df[sites].fillna(0).astype('int').to_csv('train_sessions_text.txt', 
                                               sep=' ', 
                       index=None, header=None)
test_df[sites].fillna(0).astype('int').to_csv('test_sessions_text.txt', 
                                              sep=' ', 
                       index=None, header=None)


# **Lets see the first line of one of txt files**

# In[9]:


get_ipython().system('head -1 train_sessions_text.txt')
# win equivalent
# !powershell -ExecutionPolicy Bypass "Get-Content .\train_sessions_text.txt -TotalCount 0.001kb" ;


# **Lets look into dictionary that contains urls**

# In[11]:


# Load websites dictionary
with open("../input/site_dic.pkl", "rb") as input_file:
    site_dict = pickle.load(input_file)

# Create dataframe for the dictionary
sites_dict = pd.DataFrame(list(site_dict.keys()), index=list(site_dict.values()), columns=['site'])
print(u'Websites total:', sites_dict.shape[0])
sites_dict.head()


# In[13]:


top_sites = pd.Series(train_df[sites].fillna(0).values.flatten()
                     ).value_counts().sort_values(ascending=False)
top_sites.head()


# In[14]:


alice_sites = pd.Series(train_df[train_df.target==1][sites].fillna(0).values.flatten()
                           ).value_counts().sort_values(ascending=False)
alice_sites.head()


# In[15]:


others_sites = pd.Series(train_df[train_df.target==0][sites].fillna(0).values.flatten()
                           ).value_counts().sort_values(ascending=False)
others_sites.head()


# In[16]:


alice_sites.shape, others_sites.shape, top_sites.shape


# ** Lets see that the percentage of training data w.r.t. number of websites in the dictionary **

# In[17]:


#Training data sites percentage
top_sites.shape[0]/sites_dict.shape[0]*100


# In[18]:


#Alice's sites percentage
alice_sites.shape[0]/sites_dict.shape[0]*100


# In[19]:


#Others' sites percentage
others_sites.shape[0]/sites_dict.shape[0]*100


# ** Lets see how many websites are unique to Alice in websites that he/she visits **

# In[20]:


sns.barplot(x = ['Alice', 'Alice+Others', 'Only Alice'], y=[alice_sites.shape[0], 
                                                            alice_sites.shape[0]+others_sites.shape[0]-top_sites.shape[0],
                                                            top_sites.shape[0] - others_sites.shape[0]])
plt.title('Number of websites comparison in Alice''s URL set');
plt.ylabel('# of websites');


# ** Lets see the frequencies of these websites in small dataframes **

# In[21]:


train_df_sites = train_df[sites].fillna(0).astype(int)
train_df_sites.shape


# In[22]:


top_sites_df = sites_dict.ix[top_sites.index]
top_sites_df['freq'] = top_sites
top_sites_df.head()


# In[23]:


alice_sites_df = sites_dict.ix[alice_sites.index]
alice_sites_df['freq'] = alice_sites
alice_sites_df.head()


# In[24]:


others_sites_df = sites_dict.ix[others_sites.index]
others_sites_df['freq'] = others_sites
others_sites_df.head()


# In[25]:


alice_unique_sites_index = top_sites_df.index.difference(others_sites_df.index)
alice_unique_sites_index


# In[26]:


alice_unique_sites_df = alice_sites.loc[alice_unique_sites_index]


# In[27]:


alice_unique_sites_df = pd.DataFrame(alice_unique_sites_df, columns=['freq'])
alice_unique_sites_df['site'] = sites_dict.ix[alice_unique_sites_df.index]


# In[28]:


alice_unique_sites_df.head()


# In[29]:


alice_unique_sorted_urls = alice_unique_sites_df.sort_values(by='freq',ascending=False)
alice_unique_sorted_urls.head(5)


# ** Some feature ideas: **

# In[30]:


df = train_df
hour = df['time1'].apply(lambda ts: ts.hour)
df['hour'] = df['time1'].apply(lambda ts: ts.hour)
df['morning'] = ((hour >= 7) & (hour <= 11)).astype('int')
df['noon'] = ((hour >= 12) & (hour <= 13)).astype('int')
df['afternoon'] = ((hour >= 14) & (hour <= 18)).astype('int')
df['day'] = ((hour >= 12) & (hour <= 18)).astype('int')
df['evening'] = ((hour >= 19) & (hour <= 23)).astype('int')
df['late_evening'] = ((hour >= 21) & (hour <= 23)).astype('int')
df['night'] = ((hour >= 0) & (hour <= 6)).astype('int')
df['early_night'] = ((hour >= 0) & (hour <= 2)).astype('int')
df['late_night'] = ((hour >= 3) & (hour <= 6)).astype('int')
weekday = df['time1'].apply(lambda ts: ts.dayofweek)
df['weekday'] = df['time1'].apply(lambda ts: ts.dayofweek)
df['weekend'] = ((weekday >= 5) & (weekday <= 6)).astype('int')
df['weekdays'] = (weekday <= 4).astype('int')
df['years'] = df['time1'].apply(lambda ts: ts.year)
df['weeks'] = df['time1'].apply(lambda ts: 100 * ts.year + ts.week)

# and soo on...


# ** Lets see some features in the seaborn library's factorplots. The more the difference between Alice & Intruder/Others, the better! **

# ** Intruders start hours: **

# In[31]:


df_uniques = pd.melt(frame=df[df['target']==0], value_vars=['hour'])

df_uniques = pd.DataFrame(df_uniques.groupby(['variable', 
                                              'value'])['value'].count()) \
    .sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}) \
    .reset_index()

sns.factorplot(x='variable', y='count', hue='value', 
               data=df_uniques, kind='bar', size=6);


# ** Alice start hours: **

# In[32]:


df_uniques = pd.melt(frame=df[df['target']==1], value_vars=['hour'])

df_uniques = pd.DataFrame(df_uniques.groupby(['variable', 
                                              'value'])['value'].count()) \
    .sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}) \
    .reset_index()

sns.factorplot(x='variable', y='count', hue='value', 
               data=df_uniques, kind='bar', size=6);


# ** Define a function for ease to see other features [Intruder plot first, Alice second] **

# In[33]:


def snsplot(df, feature):
    #Intruder's data
    df_uniques = pd.melt(frame=df[df['target']==0], value_vars=[feature])
    df_uniques = pd.DataFrame(df_uniques.groupby(['variable', 
                                              'value'])['value'].count()) \
    .sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}) \
    .reset_index()

    sns.factorplot(x='variable', y='count', hue='value', 
               data=df_uniques, kind='bar', size=6);
    plt.title('Intruder')
    
    # Now plot Alice's data
    df_uniques = pd.melt(frame=df[df['target']==1], value_vars=[feature])
    df_uniques = pd.DataFrame(df_uniques.groupby(['variable', 
                                              'value'])['value'].count()) \
    .sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}) \
    .reset_index()

    sns.factorplot(x='variable', y='count', hue='value', 
               data=df_uniques, kind='bar', size=6);
    plt.title('Alice')


# In[34]:


snsplot(df,'weekday')


# ** Weekday doesn't seem to be giving so much differentiating details but still you can get some ideas about the some days (try and see in your probability scores) **

# In[35]:


snsplot(df,'weekdays')


# ** Likewise, differentiating between weekdays and weekend may be useful (graphs look somewhat more inclined to weekdays for Alice. [Note: Weekend is index 5 and 6. Friday night is not considered as weekend as in English. The value 0 means weekend) **

# In[36]:


snsplot(df,'years')


# ** What about weeks and the things that they tells us **

# In[37]:


snsplot(df,'weeks')


# **From this point on, you can analyze many features with this sns factorplot function, Good luck! [Appreciate an upvote of yours if you like it, Thanks!]**
