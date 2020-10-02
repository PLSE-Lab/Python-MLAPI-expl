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

# Through this notebook, I made experiment about whether or not following fastai's tools are effective for training
#     1. fit_one_cycle
#     2. min_grad_lr
#    
# But I couldn't make any insightful conclusion. OMG

# # Tabular models

# In[ ]:


from fastai.tabular import *


# In[ ]:


path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')

dep_var = 'salary'
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
cont_names = list((set(df.columns) - set(cat_names) - set([dep_var])))
# same as cont_names = ['age', 'fnlwgt', 'education-num', 'hours-per-week', 'capital-gain', 'capital-loss']
procs = [FillMissing, Categorify, Normalize]


# In[ ]:


# Following are made in fastai notebook, but some of columns are missing (e.g. native-country, sex, hours-per-week etc..)
# Missing columns actually affect result of training in worse way
# cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
# cont_names = ['age', 'fnlwgt', 'education-num']


# In[ ]:


test = TabularList.from_df(df.iloc[800:1000].copy(), cat_names, cont_names, path=path)


# In[ ]:


data = (TabularList.from_df(df, cat_names, cont_names, procs, path=path)
                   .split_by_idx(list(range(800,1000)))
                   .label_from_df(dep_var)
                   .add_test(test)
                   .databunch())


# In[ ]:


# Experiments
learn1 = tabular_learner(data, [200,100], metrics=accuracy) # fit           without min_grad_lr
learn2 = tabular_learner(data, [200,100], metrics=accuracy) # fit           with    min_grad_lr
learn3 = tabular_learner(data, [200,100], metrics=accuracy) # fit_one_cycle without min_grad_lr
learn4 = tabular_learner(data, [200,100], metrics=accuracy) # fit_one_cycle with    min_grad_lr


# In[ ]:


learn2.lr_find()
learn2.recorder.plot(suggestion=True)
lr2 = learn2.recorder.min_grad_lr;

learn4.lr_find()
learn4.recorder.plot(suggestion=True)
lr4 = learn4.recorder.min_grad_lr;

lr2, lr4


# In[ ]:


learn1.fit(5, 1e-2)


# In[ ]:


learn2.fit(5, lr2)


# In[ ]:


learn3.fit_one_cycle(5, 1e-2)


# In[ ]:


learn4.fit_one_cycle(5, lr4)


# It is difficult to make any conclusion as to what's best at this point...

# ## Inference

# In[ ]:


learn.predict(df.iloc[0])

