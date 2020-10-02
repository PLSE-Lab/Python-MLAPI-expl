#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('ls ../input')


# ### the training set for development on local machine

# In[ ]:


import pandas as pd
df = pd.read_csv("../input/train.csv")
df= df.loc[:50000]
df.to_csv("dev_toxic.csv", index=False)


# ### test set for development

# In[ ]:


test = pd.read_csv("../input/test.csv")
print("the shape is {}".format(test.shape))


# In[ ]:


test= test.loc[:25000]
test.to_csv("test_toxic.csv", index=False)


# In[ ]:




