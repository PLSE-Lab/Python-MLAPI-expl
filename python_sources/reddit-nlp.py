#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re
import sqlite3
from sklearn.feature_extraction.text import TfidfTransformer


# In[ ]:


sql_conn = sqlite3.connect('../input/database.sqlite')


# In[ ]:


# find subreddits with most comments (top 5)
comm_counts = pd.read_sql("SELECT subreddit,COUNT(subreddit) as ncomms FROM May2015 GROUP BY subreddit ORDER BY ncomms DESC LIMIT 5", sql_conn)
comm_counts


# In[ ]:


df = pd.read_sql("SELECT subreddit,body,score FROM May2015 WHERE subreddit IN " + str(tuple(comm_counts.subreddit)) + " ORDER BY subreddit DESC LIMIT 10000",sql_conn)

# class imbalance here:
comm_counts.ncomms/sum(comm_counts.ncomms)


# In[ ]:


text = ''
for i in xrange(len(df.body)):
    text += df.body[i] + ' ' 


# In[ ]:


text = df.body.str_cat(sep = ' ')


# In[ ]:



text = re.sub('\n', '', text)
text = re.sub(r'([^\s\w]|_)+', '', text))


# In[ ]:


vec, clf = TfidfVectorizer(min_df=5), LogisticRegression(C=1.25)
td_matrix = csr_matrix(vec.fit_transform(corpus).toarray())
labels = [1]*sar_lmt+[-1]*srs_lmt
X_train, X_test, y_train, y_test = train_test_split(td_matrix, labels, 
                                   test_size=0.33, random_state=42)


# In[ ]:




