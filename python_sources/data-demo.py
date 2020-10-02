#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns


# In[ ]:


df = pd.read_csv("../input/imdb-dataset-of-50k-movie-translated-urdu-reviews/urdu_imdb_dataset.csv")


# In[ ]:


df.head()


# In[ ]:


sns.countplot(df["sentiment"]);


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:




