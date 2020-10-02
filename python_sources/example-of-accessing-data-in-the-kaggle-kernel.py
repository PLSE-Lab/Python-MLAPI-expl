#!/usr/bin/env python
# coding: utf-8

# ## Check the data in kernel
# You can use `os.listdir()` or  linux command (by adding a `!`) to check the data.

# In[ ]:


import os

print(os.listdir('../input/'))


# In[ ]:


get_ipython().system('ls ../input')


# ---
# ## Read data example

# In[ ]:


import pandas as pd

sampleSubmission = pd.read_csv('../input/sampleSubmission.csv')
sampleSubmission.head()


# ---
# ## Output data example

# In[ ]:


sampleSubmission.to_csv('./my_output.csv', index=False)
print('output ok!')


# ---
# ## Double check

# In[ ]:


print(os.listdir('../input/'))


# In[ ]:


print(os.listdir('./'))


# ---
# ## Remember to commit your kernel!
# After you finish your notebook or script on kernel, you have to `Commit` it to generate the kernel result. You can find your output file in the `Output` bookmark of your kernel result. And you can submit the result by the `Submit to Competition` button directly.

# In[ ]:




