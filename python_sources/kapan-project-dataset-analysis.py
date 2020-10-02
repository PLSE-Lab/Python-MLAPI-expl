#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Read Dataset

# In[ ]:


dataset = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv', delimiter=',', encoding='latin-1')


# ### Data of spam dataset
# 
# > *As an example, the top 10 data of the dataset is shown. We can change the number of data by changing the number 10 within the head () function*

# In[ ]:


dataset.head(10)


# ### Unnamed Columns
# 
# > *Let's remove NaN valuable and unnamed columns*

# In[ ]:


dataset.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)


# > *After removing the NaN valued and unnamed columns, our dataset is as follows*

# In[ ]:


dataset.head(10)


# ## Dataset Info
# 
# > Columns Details 
# * v1: Shows the types of messages
# * v2: messages
# * There are 5572 values
# * No null values in columns v1 and v2 (Total 5572)

# In[ ]:


dataset.info()


# ### Types Of Messages
# 
# > *The graph of HAM and SPAM numbers is given below*

# In[ ]:


sns.countplot(dataset.v1)
plt.xlabel('Labels')
plt.title('Number Of Spam & Ham')
plt.show()


# ### Types Of Messages
# 
# > *The pie graph of HAM and SPAM numbers is given below*

# In[ ]:


ham_count  = 0
spam_count = 0

for data in dataset.v1:
    if data == "ham":
        ham_count = ham_count + 1
    else:
        spam_count = spam_count + 1


# In[ ]:


slice_value = [ham_count, spam_count]
slice_title = ["Ham", "Spam"]
slice_color = ["g", "r"]

plt.pie(slice_value,
       labels=slice_title,
       colors=slice_color)

plt.title("Number Of Ham & Spam")
plt.show()

