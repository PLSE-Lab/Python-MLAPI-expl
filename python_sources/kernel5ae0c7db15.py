#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


file_path = '../input/tv-shows-on-netflix-prime-video-hulu-and-disney/tv_shows.csv'
my_data = pd.read_csv(file_path)


# In[ ]:


my_data.head(15)


# In[ ]:


import pandas as pd
import numpy as np
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


sns.barplot(x = my_data.index,y = my_data['IMDb'])
plt.figure(figsize = (14,6))


# In[ ]:


#str(my_data['age'])
sns.swarmplot(x = my_data['Age'],y = my_data['IMDb'])


# In[ ]:


print(my_data.Year.max())
print(my_data.Year.min())


# > ## <a> we have a movie list of 1901 to 2020 </a>

# In[ ]:


my_data.head()

