#!/usr/bin/env python
# coding: utf-8

# ## Tensorflow 2.2
# Not sure if there was an easier way to do this or if the 2.2 packages are already installed on a kernel someplace, but this dataset installs the packages required to run tensorflow 2.2. I needed to run 2.2 because I was training on a tpu with a keras based model and couldn't save the model with 2.1 to get it into kaggle.

# In[ ]:


get_ipython().system('pip install /kaggle/input/tf-nightly-220-dev20200311/h5py-2.10.0-cp36-cp36m-manylinux1_x86_64.whl')
get_ipython().system('pip uninstall tensorflow -y')
get_ipython().system('pip uninstall tensorboard -y')
get_ipython().system('pip install /kaggle/input/tf-nightly-220-dev20200311/gast-0.3.3-py2.py3-none-any.whl')
get_ipython().system('pip install /kaggle/input/tf-nightly-220-dev20200311/tf_estimator_nightly-2.1.0.dev2020031101-py2.py3-none-any.whl')
get_ipython().system('pip install /kaggle/input/tf-nightly-220-dev20200311/tensorboard_plugin_wit-1.6.0.post2-py3-none-any.whl')
get_ipython().system('pip install /kaggle/input/tf-nightly-220-dev20200311/gviz_api-1.9.0-py2.py3-none-any.whl')
get_ipython().system('pip install /kaggle/input/tf-nightly-220-dev20200311/tb_nightly-2.2.0a20200310-py3-none-any.whl')
get_ipython().system('pip install /kaggle/input/tf-nightly-220-dev20200311/scipy-1.4.1-cp36-cp36m-manylinux1_x86_64.whl')
get_ipython().system('pip install /kaggle/input/tf-nightly-220-dev20200311/astunparse-1.6.3-py2.py3-none-any.whl')
get_ipython().system('pip install /kaggle/input/tf-nightly-220-dev20200311/tf_nightly-2.2.0.dev20200311-cp36-cp36m-manylinux2010_x86_64.whl')


# In[ ]:


import tensorflow as tf
print(tf.version.VERSION)

