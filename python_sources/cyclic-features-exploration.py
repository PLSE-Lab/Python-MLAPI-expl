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


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nimport pandas as pd\nimport numpy as np\n\n# Load data\ntrain = pd.read_csv('../input/cat-in-the-dat/train.csv')\ntest = pd.read_csv('../input/cat-in-the-dat/test.csv')\n\nprint(train.shape)\nprint(test.shape)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Subset\ntarget = train['target']\ntrain_id = train['id']\ntest_id = test['id']\ntrain.drop(['target', 'id'], axis=1, inplace=True)\ntest.drop('id', axis=1, inplace=True)\n\nprint(train.shape)\nprint(test.shape)")


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(train['month'], target)
plt.title("Mean target balance")
f, ax = plt.subplots(1, 2, figsize=(12, 3))
sns.countplot(train['month'], ax=ax[0])
ax[0].set_title("Train")
sns.countplot(test['month'], ax=ax[1])
ax[1].set_title("Test")


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(train['day'], target)
plt.title("Mean target balance")
f, ax = plt.subplots(1, 2, figsize=(12, 3))
sns.countplot(train['day'], ax=ax[0])
ax[0].set_title("Train")
sns.countplot(test['day'], ax=ax[1])
ax[1].set_title("Test")

