#!/usr/bin/env python
# coding: utf-8

# In[51]:


import subprocess
from ast import literal_eval

def run(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    out, err = process.communicate()
    print(out.decode('utf-8').strip())


# In[49]:


print('# CPU')
run('cat /proc/cpuinfo | egrep -m 1 "^model name"')
run('cat /proc/cpuinfo | egrep -m 1 "^cpu MHz"')
run('cat /proc/cpuinfo | egrep -m 1 "^cpu cores"')


# In[43]:


print('# RAM')
run('cat /proc/meminfo | egrep "^MemTotal"')


# In[62]:


print('# OS')
run('uname -a')


# In[75]:


print('# GPU')
run('lspci | grep VGA')

