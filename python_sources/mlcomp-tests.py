#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' python ../input/mlcomp/mlcomp/mlcomp/setup.py')
get_ipython().system(' pip install pytest-xdist')


# In[ ]:


get_ipython().system(' cd /opt/conda/lib/python3.6/site-packages/mlcomp/ && pytest -v --forked --numprocesses=auto')

