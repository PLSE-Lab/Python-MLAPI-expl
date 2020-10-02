#!/usr/bin/env python
# coding: utf-8

# ### **In this kernel we'll take a brief look at the competition's dataset and try to make some naive conclusions.**

# In[ ]:


import pandas as pd
import plotly.express as px


# * Loading train and test sets, and checking some infos:

# In[ ]:


train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")


# In[ ]:


print(train_df.shape)
print(test_df.shape)


# In[ ]:


train_df.head()


# * Checking target distribution:

# In[ ]:


data = train_df
fig = px.pie(data, values='id', names='target')
fig.show()


# * Checking the lenght distribution of tweets:

# In[ ]:


data = train_df['text'].apply(lambda x: len(x))
fig = px.histogram(data, x="text")
fig.show()


# Looking at the distribution, we can easily check that the distribution is highly right skewed. The highest frequency is of tweets within 136~139 characters.
# 
# Now, let's see if there is some diference by target:

# In[ ]:


train_df['len'] = train_df['text'].apply(lambda x: len(x))

fig = px.histogram(train_df, x="len", y="target", color='target', barmode='group', height=700)
fig.show()


# Well, something possibly interesting here. As we can see, tweets up to aproximate lenght of 70 seems to have a tendency for being Not Disaster tweets (red ones)!
# 
# In some sense I believe this follows what is expected guess: When talking about a disaster, people usually use more words than just a comment about something not so important.
