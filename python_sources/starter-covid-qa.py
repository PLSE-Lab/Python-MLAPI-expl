#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import plotly.express as px


# In[ ]:


community = pd.read_csv('/kaggle/input/covidqa/community.csv')
multilingual = pd.read_csv('/kaggle/input/covidqa/multilingual.csv')
news = pd.read_csv('/kaggle/input/covidqa/news.csv')


# ## Community

# In[ ]:


community.head()


# In[ ]:


community['site'] = community.url.str.replace(".stackexchange.com", "")


# In[ ]:


px.histogram(community, x='site', color='source')


# ## Multilingual

# In[ ]:


multilingual.head()


# In[ ]:


px.histogram(multilingual, x='language', color='source')


# ## News

# In[ ]:


news.head()


# In[ ]:


fig = px.histogram(news, x='source', color='url')
fig.update_layout(showlegend=False)
fig.show()

