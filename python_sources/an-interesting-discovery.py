#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
pd.set_option('display.max_columns', 50) 
import numpy as np
import seaborn as sns


# In[ ]:


train = pd.read_csv('../input/cat-in-the-dat/train.csv')
test = pd.read_csv('../input/cat-in-the-dat/test.csv')

train.shape,test.shape


# In[ ]:


train.head()


# # A Magical Feature

#     Here, ord_1 feature depict the level of Kagglers.
#     
#     Grandmaster
#     Master
#     Expert
#     Contributor
#     Novice

# As we all know Kaggle Platform has three positions like Competition, Kernel and Discussion. And all three positions have individual levels.

# <table width="400">
#     <tr>
#         <th height="40">Competition</th>
#         <th>Kernel</th>
#         <th>Discussion</th>
#     </tr>
#     <tr>
#         <td height="30">GrandMaster</td>
#         <td>GrandMaster</td>
#         <td>GrandMaster</td>
#     </tr>
#     <tr>
#         <td height="30">Master</td>
#         <td>Master</td>
#         <td>Master</td>
#     </tr>
#     <tr>
#         <td height="30">Expert</td>
#         <td>Expert</td>
#         <td>Expert</td>
#     </tr>
#     <tr>
#         <td height="30">Contributor</td>
#         <td>Contributor</td>
#         <td>Contributor</td>
#     </tr>
#     <tr>
#         <td height="30">Novice</td>
#         <td>Novice</td>
#         <td>Novice</td>
#     </tr>
#     </table>

# using this basic knowledge, there may be one hidden feature exist which indicate this three position Competition, Kernel and Discussion.

# However, the data only have two features "ord_0" and "nom_0" which have 3 unique values. 
#     
# Competiotion, Kernel and Discussion is not look like a ordinal feature. Hence,
# Let's consider feature nom_0, it may indicate three positions.

# In[ ]:


train['nom_0'] = train['nom_0'].astype(str)
test['nom_0'] = test['nom_0'].astype(str)

train['nom0__ord1'] = train[['nom_0','ord_1']].apply(''.join, axis=1)
test['nom0__ord1'] = test[['nom_0','ord_1']].apply(''.join, axis=1)


# In[ ]:


train.head()


#     Let's apply one-hot encoding to this new feature

# In[ ]:


one_hot = pd.get_dummies(train['nom0__ord1'])
one_hot.shape


# In[ ]:


one_hot.head()


# If you found this information helpful, please let me know in the comment and upvote the notebook.

# In[ ]:




