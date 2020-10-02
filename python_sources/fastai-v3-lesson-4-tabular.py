#!/usr/bin/env python
# coding: utf-8

# [Lesson Video Link](https://course.fast.ai/videos/?lesson=4)
# 
# [Lesson resources and updates](https://forums.fast.ai/t/lesson-4-official-resources-and-updates/30317)
# 
# [Lesson chat](https://forums.fast.ai/t/lesson-4-in-class-discussion/30318/12)
# 
# [Further discussion thread](https://forums.fast.ai/t/lesson-4-advanced-discussion/30319)
# 
# Note: This is a mirror of the FastAI Lesson 4 Nb. 
# Please thank the amazing team behind fast.ai for creating these, I've merely created a mirror of the same here
# For complete info on the course, visit course.fast.ai

# # Tabular models

# In[1]:


from fastai.tabular import *


# Tabular data should be in a Pandas `DataFrame`.

# In[2]:


path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')


# In[3]:


dep_var = 'salary'
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [FillMissing, Categorify, Normalize]


# In[4]:


test = TabularList.from_df(df.iloc[800:1000].copy(), path=path, cat_names=cat_names, cont_names=cont_names)


# In[5]:


data = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_idx(list(range(800,1000)))
                           .label_from_df(cols=dep_var)
                           .add_test(test)
                           .databunch())


# In[12]:


print(df.iloc[0])
data.show_batch(rows=10)


# In[7]:


learn = tabular_learner(data, layers=[200,100], metrics=accuracy)


# In[8]:


learn.fit(1, 1e-2)


# ## Inference

# In[9]:


row = df.iloc[0]


# In[10]:


learn.predict(row)


# In[ ]:




