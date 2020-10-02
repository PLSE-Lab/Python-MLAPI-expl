#!/usr/bin/env python
# coding: utf-8

# ## The (Obigatory) Word Cloud
# So, what are they talking about? In English, at least....

# In[ ]:


import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

df = pd.read_csv('../input/tweets.csv')

junk = re.compile("al|RT|\n|&.*?;|http[s](?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)*")
tweets = [junk.sub(" ", t) for t in df.tweets]

vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=.5)
tfv = vec.fit_transform(tweets)

terms = vec.get_feature_names()
wc = WordCloud(height=1000, width=1000, max_words=1000).generate(" ".join(terms))

plt.figure(figsize=(10, 10))
plt.imshow(wc)
plt.axis("off")
plt.show()


# ## A Quick and Dirty Topic Model
# Sounds unpleasant. Let's try a topic model to get some more granularity.

# In[ ]:


from sklearn.decomposition import NMF
nmf = NMF(n_components=10).fit(tfv)
for idx, topic in enumerate(nmf.components_):
    print("Topic #%d:" % idx)
    print(" ".join([terms[i] for i in topic.argsort()[:-10 - 1:-1]]))
    print("")


# ## Top 10 Users and their Topic Distributions
# 
# If we have topics, we might as well see what the top ten users are into. Evidently Uncle_SamCoCo likes to talk about Al Qaida (?)....

# In[ ]:


import numpy as np
from matplotlib import style

style.use('bmh')

df['topic'] = np.argmax(nmf.transform(vec.transform(tweets)), axis=1)
top10_users = df[df.username.isin(df.username.value_counts()[:10].keys().tolist())]
pd.crosstab(top10_users.username, top10_users.topic).plot.bar(stacked=True, figsize=(16, 10), colormap="coolwarm")


# In[ ]:




