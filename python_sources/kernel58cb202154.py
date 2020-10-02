#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('nvidia-smi')
import os.path
get_ipython().system('apt-get -y install boinc boinc-client')
get_ipython().system('cp /usr/bin/boinc /kaggle/working')
get_ipython().system('cp /usr/bin/boinccmd /kaggle/working')
if not os.path.exists('/kaggle/working'):
  get_ipython().system('mkdir -p /kaggle/working')
if not os.path.exists('/kaggle/working/1'):
  get_ipython().system('boinc --attach_project "http://milkyway.cs.rpi.edu/milkyway/" "6d70174aa0ed0e99bfa9773a0564fc48"')
else:
  get_ipython().system('boinc')

