#!/usr/bin/env python
# coding: utf-8

# # Make requirement.txt

# In[ ]:


req = """opencv-contrib-python==3.4.2.17
opencv-python==3.4.2.17
Pillow==6.2.0
progressbar2
keras_applications
"""

get_ipython().system('echo {repr(req)} > requirements.txt')
get_ipython().system('cat ./requirements.txt')


# # Download 
# - Run with internet "on"

# In[ ]:


get_ipython().system('mkdir dep')
get_ipython().run_line_magic('cd', 'dep')
get_ipython().system('pip download -r ../requirements.txt')


# In[ ]:


get_ipython().system('mkdir /kaggle/working/github')
get_ipython().run_line_magic('cd', '/kaggle/working/github')
get_ipython().system('git clone https://github.com/cocodataset/cocoapi.git')
get_ipython().system('git clone https://github.com/xuannianz/EfficientDet.git')


# - Next, make a new dataset from the output

# # copy and make tar.gz from directories
# `!cp -r /kaggle/input/efficientdet-requirements/dep /kaggle/working/dep
# %cd /kaggle/working/dep
# !find . -maxdepth 1 -mindepth 1 -type d -exec sh -c 'cd "$1/$(basename "$1")" && tar zcvf ../../"$1".tar.gz .' sh {} ';'`

# # install
# `!pip install --no-index --find-links /kaggle/working/dep/ -r /kaggle/working/requirements.txt`
# 

# In[ ]:




