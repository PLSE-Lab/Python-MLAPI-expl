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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
get_ipython().run_line_magic('pylab', 'inline')


# In[ ]:


df = pd.read_csv("../input/train.csv")


# In[ ]:


df


# In[ ]:


df.describe()


# In[ ]:





# In[ ]:


df.Survived.value_counts()


# In[ ]:


df.Sex.value_counts().plot(kind='bar')


# In[ ]:


df[df.Sex=='female']


# In[ ]:


df[df.Age.isnull()]


# In[ ]:


df.describe()


# In[ ]:


df.Fare.hist(bins=5)


# In[ ]:


df[df.Fare==0]


# In[ ]:


df[df.Sex=='male'].Survived.value_counts().plot(kind='bar')


# In[ ]:


df[(df.Age<25)&(df.Sex=='female')].Survived.value_counts().plot(kind='bar')


# In[ ]:


df[(df.Age<25)&(df.Sex=='male')].Survived.value_counts().plot(kind='bar')


# In[ ]:


df[df.Age.isnull()]


# In[ ]:




