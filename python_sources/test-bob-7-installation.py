#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' ls -F /kaggle/input/python-bob-700/conda')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'conda install --offline --yes /kaggle/input/python-bob-700/conda/*')


# In[ ]:


import bob.io.matlab


# In[ ]:


get_ipython().system(' ls /opt/conda/lib/python3.6/site-packages/bob/db/atnt/data')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'export ATNT_DATABASE_DIRECTORY=/opt/conda/lib/python3.6/site-packages/bob/db/atnt/data\nnosetests -sv bob --exclude=test_extensions --exclude=test_driver')


# In[ ]:




