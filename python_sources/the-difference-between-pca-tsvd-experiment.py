#!/usr/bin/env python
# coding: utf-8

# **In this little experiment, I will build 2 different sparse matrix.**
# 
# **You can PCA and tSVD give us different performance on the sparse matrix hope you gain some intuitive though from the below results.**

# In[ ]:


from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('precision', '3')


# **Sparse Matrix 1**

# In[ ]:


row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
data = csr_matrix((data, (row, col)), shape=(10, 10)).toarray()
print(data)


# In[ ]:


pca = PCA(n_components=5)
pca.fit_transform(data)


# In[ ]:


tsvd = TruncatedSVD(n_components=5)
tsvd.fit_transform(data)


# **Sparse Matrix 2**
# 
# **Note Please be careful!!!! At this case, I add one value '10' at the position (4,4) in the matrix, to see how PCA and tSVD works on it**

# In[ ]:


row = np.array([0, 0, 1, 2, 2, 2, 4])
col = np.array([0, 2, 2, 0, 1, 2, 4])
data = np.array([1, 2, 3, 4, 5, 6, 10])
data = csr_matrix((data, (row, col)), shape=(10, 10)).toarray()
print(data)


# In[ ]:


pca = PCA(n_components=5)
pca.fit_transform(data)


# In[ ]:


tsvd = TruncatedSVD(n_components=5)
tsvd.fit_transform(data)


# In[ ]:




