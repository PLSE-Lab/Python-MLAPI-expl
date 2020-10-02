#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import pyarrow.parquet as pq
import os


# In[ ]:


import pandas as pd
import pyarrow.parquet as pq
import os


# In[ ]:


os.listdir('../input')


# In[ ]:


train = pq.read_pandas('../input/train.parquet').to_pandas()


# In[ ]:


train.info()


# In[ ]:


train.columns[:5]


# In[ ]:


subset_train = pq.read_pandas('../input/train.parquet', columns=[str(i) for i in range(5)]).to_pandas()


# In[ ]:


subset_train.info()


# In[ ]:


subset_train.head()


# In[ ]:




