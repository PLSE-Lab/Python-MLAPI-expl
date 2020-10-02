#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Exploratory data analysis of rockyou.txt

# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import hashlib

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))


# In[ ]:


get_ipython().run_cell_magic('time', '', '#dataframe = pd.read_fwf(\'../input/rockyou.txt\', widths=24)\n\ndataframe = pd.read_csv(\'../input/rockyou.txt\',\n                        delimiter = "\\n", \n                        header = None, \n                        names = ["Passwords"],\n                        encoding = "ISO-8859-1")')


# In[ ]:


get_ipython().run_cell_magic('time', '', "# dataframe.to_csv('../input/rockyou.csv')\n# OSError: [Errno 30] Read-only file system: '../input/rockyou.csv'")


# In[ ]:


dataframe.info()


# In[ ]:


dataframe.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "dataframe['MD5'] = [hashlib.md5(str.encode(str(i))).hexdigest() \n                    for i in dataframe['Passwords'].fillna(0).astype(str)]")


# In[ ]:


dataframe.head()


# In[ ]:


dataframe.info()


# In[ ]:


# Drop duplicate password
dataframe.drop_duplicates(subset=['Passwords'], keep=False, inplace=True)


# In[ ]:


dataframe.info()


# In[ ]:


get_ipython().run_cell_magic('time', '', "## dataframe['Passwords'].value_counts()\n# get indexes\n## dataframe['Passwords'].value_counts().index.tolist()\n# get values of occurrences\n## dataframe['Passwords'].value_counts().values.tolist()")


# In[ ]:


# delete all rows with password over 20 letters and less than 3
clutter = dataframe[ (dataframe['Passwords'].str.len() >= 20) 
                   | (dataframe['Passwords'].str.len() <= 3) ].index
dataframe.drop(clutter, inplace=True)


# In[ ]:


print (dataframe['Passwords'].str.len().value_counts())


# In[ ]:


dataframe.info()


# In[ ]:


dataframe = dataframe.reset_index(drop=True)


# In[ ]:


dataframe.info()


# In[ ]:




