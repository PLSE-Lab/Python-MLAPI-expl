#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import json 
import os
import gc

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[2]:


print(os.listdir("../input"))


# In[3]:


sub = pd.read_csv('../input/sample_submission.csv')


# In[4]:


with open('../input/train_annotations.json/train_annotations.json','r') as anno_train:
    train = json.load(anno_train)
    
with open('../input/test_annotations.json/test_annotations.json','r') as anno_test:
    test = json.load(anno_test)


# In[5]:


train.keys()


# In[6]:


test.keys()


# In[7]:


test_df = pd.DataFrame() 
test_df = test_df.append(test['images'], ignore_index=True)


# In[8]:


train_df = pd.DataFrame() 
train_df = train_df.append(train['images'], ignore_index=True)
train_df_anno = pd.DataFrame() 
train_df_anno = train_df_anno.append(train['annotations'], ignore_index=True)
train_df['category_id'] = train_df_anno['category_id']
del train_df_anno
gc.collect()


# In[9]:


test_df.head()


# In[10]:


train_df.head()


# In[11]:


sub.head()


# In[12]:


print(len(train_df)/len(test_df))


# > Train:Test = 473438/59141 = 8

# In[13]:


category_numbers =  [0]*(np.max(train_df['category_id'])+1)


# In[14]:


for i in train_df['category_id']:
    category_numbers[i]+=1


# In[22]:


plt.xlabel('class')
plt.ylabel('number of samples')
x = list(range(len(category_numbers)))
plt.bar(x[1:], category_numbers[1:])


# In[16]:


max(category_numbers[1:])


# the max number of sample in one category:1000

# In[17]:


category_numbers[0]


# the firtst class may be "the other categorie".

# In[21]:


plt.xlabel('class')
plt.ylabel('number of samples')
plt.plot(x[1:], sorted(category_numbers[1:], reverse=True))


# In[19]:


few_shot = 0
for i in category_numbers[1:]:
    if i <= 200:
        few_shot+=1


# In[20]:


few_shot/len(category_numbers)


# In[ ]:




