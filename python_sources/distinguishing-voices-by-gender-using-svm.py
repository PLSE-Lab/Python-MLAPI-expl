#!/usr/bin/env python
# coding: utf-8

# ## Fetch the Dataset into Dataframe
# 
# First of all, let's add mandatory libraries and check the dataset directory.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# And fetch dataset into dataframe by using the directory above.

# In[ ]:


df = pd.read_csv('/kaggle/input/voicegender/voice.csv')


# **Importing Necessary Libraries**
# 
# Libraries imported below are necessary for the future use.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Exploration
# 
# To benefit data as much as we can, we need a good understanding of it. So let's investigate the dataframe variable in details by using `df.info()` method.

# In[ ]:


df.info()


# There are 21 attributes and 3168 entries. All of the attributes are numeric values, excepting `label`, which is the **target** variable.
# 
# So let's take a glimpse of data to see what kind of float numbers we do have, and how target variable is defined.

# In[ ]:


df.head()


# Most of the attribute values are scaled between 0 and 1. But not all of them are. So we will be normalizing these features in the Data Preparation section.
# 
# Target feature is whether `male` or `female`. We will be binding them as 0 or 1.

# ### Correlation
# Let's check the correlation between features by using correlation matrix, and visualize it via seaborn's heatmap library

# In[ ]:


f, axis = plt.subplots(figsize = (18,18))
sns.heatmap(df.corr(), annot = False, linewidths = .4, ax = axis)
plt.show()


# In[ ]:


df.label.value_counts()


# In[ ]:


df.label = [1 if each == 'female' else 0 for each in df.label]


# In[ ]:


y = df['label'].values
x = df.drop(['label'], axis = 1)


# In[ ]:


y


# In[ ]:


x.head()


# In[ ]:


F


# 
