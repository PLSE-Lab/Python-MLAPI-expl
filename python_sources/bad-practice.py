#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import networkx as nx
from networkx.utils import cuthill_mckee_ordering


# The aim of this notebook is to get gibas features in the least amount of effor possible
# I remembered Cuthill-McKee that does all the jumbling or row and columns
# The problem is it expects a squared symmetric matrix - train is not square (c.a 4900 columns 4500 rows) and it definitely not symmetric
# 
# One could take pad the Matrix with zeros to make it square and we could make the matrix as symmetric as possible(W = np.maximum( A, A.transpose() ))
# 
# However for this adventure I am going to cut off the end columns to make it square and ignore that fact that it isn't symmetric to see if we can still get gibas features (**I suggest strongly you try the above steps**)

# In[ ]:


gibasfeatures = ["f190486d6","58e2e02e6","eeb9cd3aa","9fd594eec","6eef030c1",
                 "15ace8c9f","fb0f5dbfe","58e056e12","20aa07010","024c577b9",
                 "d6bb78916","b43a7cfd5","58232a6fb"]


# In[ ]:


train = pd.read_csv('../input/train.csv')
orig = train[train.columns[2:]].copy()
orig.head()


# In[ ]:


G = nx.from_numpy_matrix((orig.values[:,:4459])) #Make it square ignore last columns and shut eyes symmetry wise
cm = list(cuthill_mckee_ordering(G))


# In[ ]:


A = nx.adjacency_matrix(G, nodelist=cm)


# In[ ]:


cm_df = pd.DataFrame(A.todense(),columns = np.array(orig.columns)[cm], index=orig.index[cm])
cm_df.head()


# In[ ]:


featurecount = cm_df[cm_df!=0].count(axis=0)
featurecount.head()


# In[ ]:


for f in gibasfeatures:
    print(featurecount[featurecount.index==f])


# In[ ]:


featurecount[featurecount>=1447].shape #Only 56 features higher than 1447


# In[ ]:


features = featurecount[featurecount>=1447].index.ravel()


# In[ ]:


set(gibasfeatures).intersection(features)


# So we reduced the 4559 features to 56 and those 56 contain all of gibas features (Lucky they didn't appear in the chopped off columns)
# It may be complete luck that they appear but with 4559 columns to choose from we had to be very lucky to get them all in the top 60
