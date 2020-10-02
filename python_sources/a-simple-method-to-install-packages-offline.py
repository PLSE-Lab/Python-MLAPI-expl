#!/usr/bin/env python
# coding: utf-8

# # A simple method to install package that you wanna use offline

# During the game, we need a lot of packages, like numpy, pandas and so on. Most of them are pre-installed by kaggle, but some are not.
# 
# As we can't use the internet when submitting, we need to install them offline. 
# 
# Here I'll show you a simple example by installing face_recognition.

# ## First, switch internet on

# In[ ]:


get_ipython().system('pip install face_recognition')


# As you can see, the face_recognition is based on dlib and face-recognition-models. So the kernel will download them first.
# ### Download .whl file directly
# Click the link in the second line, and the .whl file for face_recognition will be downloaded in your PC.
# ### Find compiled .tar.gz file
# Installing tar.gz file is a little bit troublesome, but we can find the corresponding whl file in the `/tmp/.cache/pip/wheels/`

# In[ ]:


get_ipython().system('cp /tmp/.cache/pip/wheels/96/ac/11/8aadec62cb4fb5b264a9b1b042caf415de9a75f5e165d79a51/dlib-19.19.0-cp36-cp36m-linux_x86_64.whl /kaggle/working')
get_ipython().system('cp /tmp/.cache/pip/wheels/d2/99/18/59c6c8f01e39810415c0e63f5bede7d83dfb0ffc039865465f/face_recognition_models-0.3.0-py2.py3-none-any.whl /kaggle/working')


# Click the `Refresh` button next to the `/kaggle/working` and then you can download the .whl files

# ## Upload the .whl files to your own dataset
# Go to https://www.kaggle.com/datasets and click `New dataset`. Now you can make your own dataset.

# ## Add Data and then install offline

# In[ ]:




