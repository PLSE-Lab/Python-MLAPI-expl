#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df = pd.read_csv("/kaggle/input/nzta-crash-analysis-system-cas/Crash_Analysis_System_CAS_data.csv")')


# In[ ]:


df.columns


# In[ ]:


akl = df[df.region == "Auckland Region"]
akl.crashSeverity.value_counts()


# In[ ]:


akl.crashYear.value_counts().plot(kind="bar")


# In[ ]:


akl.weatherA.value_counts().plot(kind="bar")


# In[ ]:


akl.crashLocation1.value_counts().head(20)

