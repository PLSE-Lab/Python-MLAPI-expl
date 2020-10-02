#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/tweets.csv')
df.columns = ['Name','UName','Desc','Location','Followers','NStatus','Time','Tweets']
df['Mentions'] = df.Tweets.str.count('@')
df.index.name = "row_num"
# df['Translated'] = df.Tweets.str.contains('ENGLISH TRANSLATION:') | df.Tweets.str.contains('ENGLISH TRANSCRIPT:')
# df['Tweets'] = df.Tweets.str.replace('ENGLISH TRANSLATION:',"").str.strip()
print (df.count())
(df[8:18])


# In[ ]:


fig, ax = plt.subplots(figsize=(14,6))
ax.hist(df.NStatus, bins=np.arange(0,18000,500), label='# Status')
ax.hist(df.Followers, bins=np.arange(0,18000,200), alpha=0.9)
plt.legend()
ax.set_xlabel('Followers')
plt.show()


# In[ ]:


gr = df.Tweets.str.extractall('(\@(\w+))')
gr.index.names = ["row_num","match"]
gr = gr.join(df, how='inner')
gr = gr[['UName',1,'Mentions']]
gr['Weight'] = 1/(gr.Mentions)
gr.rename(columns={1 : "At"}, inplace=True)
gr = gr.groupby(['UName','At']).agg({"Weight":"sum"}).reset_index()
gr.sample(5)


# In[ ]:


import networkx as nx
G = nx.from_pandas_dataframe(gr, 'UName', 'At', ['Weight'])
nx.draw(G)


# In[ ]:


print (max([len(i) for i in nx.find_cliques(G)]))


# In[ ]:




