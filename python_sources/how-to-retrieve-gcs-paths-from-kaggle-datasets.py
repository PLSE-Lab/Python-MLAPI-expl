#!/usr/bin/env python
# coding: utf-8

# # How to retrieve public GCS paths from public Kaggle Datasets

# *Step 1: List the contents of a Kaggle Dataset*

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# *Step 2: Use KaggleDatasets().get_gcs_path() to retrieve public GCS paths from a public Kaggle dataset*

# In[ ]:


from kaggle_datasets import KaggleDatasets
GCS_PATH = KaggleDatasets().get_gcs_path()


# In[ ]:


get_ipython().system('gsutil ls $GCS_PATH')


# In[ ]:


get_ipython().system('gsutil version -l')


# In[ ]:




