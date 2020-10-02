#!/usr/bin/env python
# coding: utf-8

# **Sometimes you need to install custom packages in Kaggle Notebooks without internet access. You can do that uploading source/precompiled code as Kaggle Datasets.**
# 
# **In this example, we install Tensorflow 1.13.1 (GPU version) and bert-tensorflow.**

# In[ ]:


get_ipython().system('ls ../input/tensorflow1131-offline-bert')


# In[ ]:


def setup_tensorflow_1_13():
    
    # Install `tensorflow-gpu==1.13.1` from pre-downloaded wheels
    PATH_TO_TF_WHEELS = '/kaggle/input/tensorflow1131-offline-bert/tensorflow_gpu_1_13_1_with_deps_whl/tensorflow_gpu_1_13_1_with_deps_whl'
    # yes, mixing up Python code and bash is ugly. But it's handy 
    get_ipython().system('pip install --no-deps $PATH_TO_TF_WHEELS/*.whl')


# In[ ]:


get_ipython().run_cell_magic('capture', '', 'setup_tensorflow_1_13()')


# And there you go

# In[ ]:


import tensorflow as tf
print(tf.__version__)


# **Also, you can just add source code and add a corresponding path to Python path.**

# In[ ]:


get_ipython().system('ls /kaggle/input/tensorflow1131-offline-bert/bert-tensorflow-1.0.1/bert-tensorflow-1.0.1/')


# In[ ]:


import sys
sys.path.append('/kaggle/input/tensorflow1131-offline-bert/bert-tensorflow-1.0.1/bert-tensorflow-1.0.1/')


# In[ ]:


from bert import modeling

