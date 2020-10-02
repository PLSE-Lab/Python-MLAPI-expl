#!/usr/bin/env python
# coding: utf-8

# # Kernel demonstrates how to use AutoViz, a new Python Library that automatically visualizes any data set, any size with oneline of code.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().system('pip install autoviz')


# In[ ]:


from autoviz.AutoViz_Class import AutoViz_Class


# In[ ]:


AV = AutoViz_Class()


# In[ ]:


url1 = '../input/ram-reduce/reduce_train.csv'
url2 = '../input/ram-reduce/reduce_test.csv'
reduce_train = pd.read_csv(url1,index_col=None)
reduce_test = pd.read_csv(url2,index_col=None)
print(reduce_train.shape,reduce_test.shape)


# In[ ]:


target='accuracy_group'


# In[ ]:


dft = AV.AutoViz(depVar=target, dfte=reduce_train, header=0, verbose=0,
                lowess=False,chart_format='svg',max_rows_analyzed=1500,max_cols_analyzed=30,filename='', sep=',' )


# In[ ]:




