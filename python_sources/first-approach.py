#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import tensorflow as tf


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import clear_output


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
train_df_sorted = train_df.sort_values('Target')
train_df_sorted.head()


# # Let's see if there's some patterns in it

# In[ ]:


plt.plot(train_df_sorted.Target.tolist())
plt.show()
train_df_sorted.Target.tolist().count(4)/len(train_df_sorted.Target.tolist())


# In[ ]:


plt.plot(train_df_sorted.v2a1.tolist())


# In[ ]:


plt.plot(train_df_sorted.Target.tolist())
plt.plot(np.array(train_df_sorted.hacdor.tolist())*2)


# In[ ]:


plt.plot(train_df_sorted.Target.tolist())
plt.plot(np.array(train_df_sorted.rooms.tolist())/10)


# In[ ]:


plt.plot(train_df_sorted.Target.tolist())
plt.plot(np.array(train_df_sorted.meaneduc.tolist())/20)


# In[ ]:


plt.plot(train_df_sorted.Target.tolist())
plt.plot(np.array(train_df_sorted.hhsize.tolist())/10)


# ## not really
# ### so just let us just guess

# In[ ]:


test_df = pd.read_csv('../input/test.csv')
test_id = test_df.Id.tolist()


# In[ ]:


pre = []
for i in range(len(test_id)):
    pre.append(4)
df = pd.DataFrame({'Id':test_id,'Target':pre})
df.to_csv('pre4.csv',index=False)


# In[ ]:


pre = []
for i in range(len(test_id)):
    pre.append(3)
df = pd.DataFrame({'Id':test_id,'Target':pre})
df.to_csv('pre3.csv',index=False)


# In[ ]:


pre = []
for i in range(len(test_id)):
    pre.append(2)
df = pd.DataFrame({'Id':test_id,'Target':pre})
df.to_csv('pre2.csv',index=False)


# In[ ]:


pre = []
for i in range(len(test_id)):
    pre.append(1)
df = pd.DataFrame({'Id':test_id,'Target':pre})
df.to_csv('pre1.csv',index=False)


# In[ ]:


pre = []
for i in range(len(test_id)):
    pre.append(np.random.randint(1,5))
df = pd.DataFrame({'Id':test_id,'Target':pre})
df.to_csv('prernd.csv',index=False)

