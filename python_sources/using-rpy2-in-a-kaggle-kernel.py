#!/usr/bin/env python
# coding: utf-8

# # Using R and Python in a Kaggle Kernel 
# 
# One should normally think that it is possible to use `R` within `python` on a Kaggle Kernel by simply typing `!pip install rpy2`. However this outputs the following error. 

# In[ ]:


get_ipython().system('pip install rpy2')


# Notice that `R` is not installed inside the anaconda environment. So we could use the `subprocess` library to execute a conda shell command, that is, 

# In[ ]:


import subprocess
subprocess.run('conda install -c conda-forge r-base', shell=True)


# Now, we just simply type the usual command to install `rpy2`

# In[ ]:


get_ipython().system('pip install rpy2')


# and verify by importing the module. 

# In[ ]:


import rpy2 

