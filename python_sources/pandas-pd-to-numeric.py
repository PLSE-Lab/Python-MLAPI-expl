#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Pandas -  pd.to_numeric

# this notebook is made to share with you the benifit of pandas [pd.to_numeric](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_numeric.html[](http://) method.
# which will convert argument to a numeric type.
# 
# 
# this method is helpful specially for data validation section.
# imagine we have a column that supposed to be numeric, but for some reason the data came from API, CSV or data base with the type of string, and that column includes many mistakes and missing data.
# this method can help us to validate the date and to convert each value in that cloumn to a numeric type if it correct and invalid parsing will be set as NaN.

# ## Examples

# In[ ]:


import pandas as pd


# In[ ]:


s = pd.Series(['11.0', '20', -30], name='example')
s


# In[ ]:


pd.to_numeric(s, downcast='float')


# In[ ]:


pd.to_numeric(s, downcast='signed')


# In[ ]:


s = pd.Series(['mistake', '11.0', '20', -3])
s


# In[ ]:


s = pd.to_numeric(s, errors='coerce')
s


# we can see here that mistakes will be automatically converted to NaN, and numbers will be converted to numeric values.

# In[ ]:


s.dtype


# and finally we can see that the dtype of our series is float64.

# ## Thanks

# [Ayoub Abozer](https://www.kaggle.com/ayoubabozer)
