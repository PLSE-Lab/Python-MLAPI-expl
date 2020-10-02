#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/download"))

# Any results you write to the current directory are saved as output.


# In[2]:


pd.set_option('display.float_format', lambda x: '%.1f' % x)


# In[3]:


transactions = pd.read_csv('../input/download/transactions.csv')


# In[4]:


transactions.head()


# In[5]:


ratings = pd.read_csv('../input/download/ratings.csv')
ratings.head()


# In[6]:


ratings.rating.value_counts()


# In[7]:


ratings.shape


# In[8]:


ratings_top = ratings[ratings.rating > 7]
ratings_top.shape


# In[9]:


def create_interaction_matrix(df,user_col, item_col, rating_col, norm= False, threshold = None):
    '''
    Function to create an interaction matrix dataframe from transactional type interactions
    Required Input -
        - df = Pandas DataFrame containing user-item interactions
        - user_col = column name containing user's identifier
        - item_col = column name containing item's identifier
        - rating col = column name containing user feedback on interaction with a given item
        - norm (optional) = True if a normalization of ratings is needed
        - threshold (required if norm = True) = value above which the rating is favorable
    Expected output - 
        - Pandas dataframe with user-item interactions ready to be fed in a recommendation algorithm
    '''
    interactions = df.groupby([user_col, item_col])[rating_col]             .sum().unstack().reset_index().             fillna(0).set_index(user_col)
    if norm:
        interactions = interactions.applymap(lambda x: 1 if x > threshold else 0)
    return interactions


# In[10]:


get_ipython().run_cell_magic('time', '', "interactions = create_interaction_matrix(df = ratings,\n                                         user_col = 'user_uid',\n                                         item_col = 'element_uid',\n                                         rating_col = 'rating', \n                                         threshold = 7)")


# In[11]:


from scipy import sparse
from lightfm import LightFM
from sklearn.metrics.pairwise import cosine_similarity

def create_user_dict(interactions):
    '''
    Function to create a user dictionary based on their index and number in interaction dataset
    Required Input - 
        interactions - dataset create by create_interaction_matrix
    Expected Output -
        user_dict - Dictionary type output containing interaction_index as key and user_id as value
    '''
    user_id = list(interactions.index)
    user_dict = {}
    counter = 0 
    for i in user_id:
        user_dict[i] = counter
        counter += 1
    return user_dict

user_dict = create_user_dict(interactions=interactions)


# In[12]:


inv_user_dict = {v: k for k, v in user_dict.items()}


# In[13]:


movie_id = list(interactions.columns)
movie_dict = {}
counter = 0 
for i in movie_id:
    movie_dict[i] = counter
    counter += 1


# In[14]:


inv_movie_dict = {v: k for k, v in movie_dict.items()}


# In[15]:


get_ipython().run_cell_magic('time', '', 'x = sparse.csr_matrix(interactions.values)')


# In[16]:


sparse.save_npz('sparse_matrix.npz', x)


# In[17]:


get_ipython().system('ls')


# In[18]:


model = LightFM(no_components= 30, loss='warp',k=15)

model.fit(x,epochs=30,num_threads = 10, verbose=True)


# In[19]:


from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score


# In[20]:


# %%time
# train_auc = auc_score(model, x, num_threads=10)

# train_auc.mean()


# In[21]:


# train_auc.mean()


# In[22]:


# %%time
# train_precision_at_k = precision_at_k(model, x, num_threads=10, k=30)

# print(train_precision_at_k.mean())


# In[23]:


ratings.head()


# In[24]:


n_items = ratings.element_uid.nunique()


# In[25]:


import json

with open('../input/download/sample_answer.json') as sa:
    sample_answer = json.loads(sa.read())


# In[26]:


sample_answer['260059']


# In[27]:


top10_20 = ratings[ratings.rating == 10].element_uid.value_counts()[0:20].index.tolist()
top910_20 = ratings[ratings.rating.isin([9, 10])].element_uid.value_counts()[0:20].index.tolist()
top8910_20 = ratings[ratings.rating.isin([8, 9, 10])].element_uid.value_counts()[0:20].index.tolist()


# In[ ]:





# In[28]:


top10_sumbission = {}
for user in sample_answer:
    top10_sumbission[str(user)] = top10_20

with open('top10_sumbission.json', 'w') as fp:
    json.dump(top10_sumbission, fp)
    
top910_sumbission = {}
for user in sample_answer:
    top910_sumbission[str(user)] = top910_20

with open('top910_sumbission.json', 'w') as fp:
    json.dump(top910_sumbission, fp)
    
top8910_sumbission = {}
for user in sample_answer:
    top8910_sumbission[str(user)] = top8910_20

with open('top8910_sumbission.json', 'w') as fp:
    json.dump(top8910_sumbission, fp)


# In[29]:


old_users = []
new_users = []

for user in sample_answer:
    if int(user) in user_dict:
        old_users.append(user)
    else:
        new_users.append(user)
print (len(old_users), len(new_users))


# In[30]:


scores = model.predict(user_dict[int('260059')], np.arange(n_items))
items_20 = np.argsort(-scores)[0:20]
items_20


# In[31]:


def get_rec_for_user(user_id_str):
    scores = model.predict(user_dict[int(user_id_str)], np.arange(n_items))
    items_20 = np.argsort(-scores)[0:20]
    return items_20


# In[34]:


for user in old_users:
    top10_sumbission[user] = get_rec_for_user(user).tolist()
    
with open('top10_sumbission_FMed.json', 'w') as fp:
    json.dump(top10_sumbission, fp)


# In[ ]:


for user in old_users:
    top910_sumbission[user] = get_rec_for_user(user).tolist()
    
with open('top910_sumbission_FMed.json', 'w') as fp:
    json.dump(top910_sumbission, fp)


# In[ ]:


for user in old_users:
    top8910_sumbission[user] = get_rec_for_user(user).tolist()
    
with open('top8910_sumbission_FMed.json', 'w') as fp:
    json.dump(top8910_sumbission, fp)

