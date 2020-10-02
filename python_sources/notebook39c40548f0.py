#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import seaborn as sns
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pkmn = pd.read_csv('../input/Pokemon.csv')
pkmn.head()

# Any results you write to the current directory are saved as output.


# In[ ]:


sns.jointplot(x="HP", y="Defense", data=pkmn);


# In[ ]:


sns.boxplot(data=pkmn);

