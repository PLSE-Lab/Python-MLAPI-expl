#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scipy.io import loadmat
import pandas as pd


# In[ ]:


mat = loadmat('../input/train_1/1_1_0.mat')


# In[ ]:


mdata = mat['dataStruct']


# In[ ]:


mtype = mdata.dtype


# In[ ]:


ndata = {n: mdata[n][0,0] for n in mtype.names}


# In[ ]:


ndata


# In[ ]:


data_headline = ndata['channelIndices']
print(data_headline)


# In[ ]:


data_headline = data_headline[0]


# In[ ]:


data_raw = ndata['data']
len(data_raw)


# In[ ]:


pdata = pd.DataFrame(data_raw,columns=data_headline)


# In[ ]:


pdata


# In[ ]:




