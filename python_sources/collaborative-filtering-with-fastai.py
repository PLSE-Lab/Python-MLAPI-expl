#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.collab import *
from fastai.tabular import *
import seaborn as sns


# In[2]:


ratings = pd.read_csv('/kaggle/input/songsDataset.csv')
ratings.head()


# In[3]:


ratings.columns = ['userID', 'songID', 'rating']


# In[4]:


len(ratings)


# In[5]:


ratings['rating'].value_counts()


# In[6]:


sns.countplot(ratings['rating'])


# In[7]:


data = CollabDataBunch.from_df(ratings, seed=42, valid_pct=0.2)


# In[8]:


data.show_batch()


# In[9]:


y_range = [0.5,5.5]


# In[10]:


learn = collab_learner(data, n_factors=50, y_range=y_range, wd=1e-1)


# In[11]:


learn.lr_find()
learn.recorder.plot(skip_end=15)


# In[12]:


learn.fit_one_cycle(5, 5e-3)


# In[14]:


learn.save('/kaggle/working/colab-50')


# In[17]:


learn.model


# In[15]:


# uncomment these lines of code to try various sizes for embeddings

# for factor in [5,10,20,30]:
#     print("results for n_factors = " + str(factor))
#     learn = collab_learner(data, n_factors=factor, y_range=y_range, wd=1e-1)
#     learn.fit_one_cycle(5, 5e-3)


# In[ ]:




