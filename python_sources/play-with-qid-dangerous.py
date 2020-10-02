#!/usr/bin/env python
# coding: utf-8

# **Thoughts:**
# 
# * Distributed evenly in first 1-2 chars
# 
# * First 3 or 4 chars of qid seems interesting (some id gives super high probability for prediction)... 
# 
# * Larger than 4 seems to be unique which is useless
# 
# * If I print out some raw data, there is not nothing really interesting. The question is how qid was generated: related to topic or time or just randomly
# 
# * Just to have some fun with qid. Be careful before using first 3 or 4 chars as feature ...

# In[ ]:


import pandas as pd
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_train.head()


# In[ ]:


df = df_train[['qid', 'question_text']].append(df_test)


# In[ ]:


df[df.duplicated(['qid'], keep=False)]


# In[ ]:


for i in range(1, 21):
    df_train['first_' + str(i)] = df_train['qid'].apply(lambda x: x[:i])
    
df_train.head()


# **First char**

# In[ ]:


df_train.groupby('first_1')['target'].agg(['count', 'mean']).reset_index().sort_values('mean', ascending=False)


# **Frist 2 chars**

# In[ ]:


df_train.groupby('first_2')['target'].agg(['count', 'mean']).reset_index().sort_values('mean', ascending=False)


# **Frist 3 chars**

# In[ ]:


df_train.groupby('first_3')['target'].agg(['count', 'mean']).reset_index().sort_values('mean', ascending=False)


# **First 4 chars**

# In[ ]:


df_train.groupby('first_4')['target'].agg(['count', 'mean']).reset_index().sort_values('mean', ascending=False)


# **First 5 chars**

# In[ ]:


df_train.groupby('first_5')['target'].agg(['count', 'mean']).reset_index().sort_values('mean', ascending=False)


# **Check the raw data**

# In[ ]:


df_2122 = df_train[df_train.first_4 == '2122']
for index, row in df_2122.iterrows():
    print(row['target'], ': ',row['question_text'])


# In[ ]:


df_623a = df_train[df_train.first_4 == '623a']
for index, row in df_623a.iterrows():
    print(row['target'], ': ',row['question_text'])


# 

# Do some prediction:

# In[ ]:


df_test['first_4'] = df_test['qid'].apply(lambda x: x[:4])
stat = df_train.groupby('first_4')['target'].agg(['mean']).reset_index()
result = pd.merge(df_test, stat, how='left', on=['first_4'])
result.head(n=100)


# 

# **Check the 20th row:**
# 
# Is a decision tree better than logistic regres...  and 0.000000

# **I don't have courage to play with it further :)**
