#!/usr/bin/env python
# coding: utf-8

# # Parallel KFold training on TPU using Pytorch Lightning
# This kernel demonstrates training K instances of a model parallely on each TPU core.

# ### Install XLA
# XLA powers the TPU support for PyTorch

# In[ ]:


get_ipython().system('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
get_ipython().system('python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev')


# In[ ]:


get_ipython().system('git clone https://github.com/lezwon/pytorch-lightning.git')


# In[ ]:


get_ipython().system('cd pytorch-lightning && git pull origin 1246_tpu_tests && git checkout 1246_tpu_tests')


# In[ ]:


get_ipython().system('pip install -r pytorch-lightning/tests/requirements.txt')


# In[ ]:


get_ipython().system('pytest pytorch-lightning/tests/models/test_tpu.py')


# In[ ]:


get_ipython().system('pytest pytorch-lightning/tests/trainer/test_trainer.py::test_tpu_choice')


# In[ ]:




