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


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#importing dataset
data= pd.read_csv("../input/tv-shows-on-netflix-prime-video-hulu-and-disney/tv_shows.csv")


# In[ ]:


data.head()


# In[ ]:


#getting some information from the data set
data.describe()


# In[ ]:


#shape of a data frame
data.shape


# In[ ]:


data.size


# In[ ]:


data.columns


# This data has various TV Shows and their ratings etc. The % from rotten tomatoes must be removed and the + from age ratings must be removed. The data has to be cleaned properly to do the proper analysis. We will try to do an overall analysis of the TV Shows and understand the trends

# In[ ]:


data.info()


# In[ ]:


#converting the Rotten tamotes column to a number by Removing %
data["Rotten Tomatoes"]= data['Rotten Tomatoes'].str.replace('%',"").astype(float)
data['Rotten Tomatoes']
#data['Rotten Tomatoes'].apply(lambda x:float(x))


# In[ ]:


#Removing the "+" sign from age rating

data["Age"] = data["Age"].str.replace("+","")


# In[ ]:


#Conveting it to numeric 

data['Age'] = pd.to_numeric(data['Age'],errors='coerce')


# In[ ]:


#Final Data
data.head()


# In[ ]:


data.nunique()


# In[ ]:


data.isnull().sum()


# In[ ]:


data2= data.dropna()


# In[ ]:


data2


# In[ ]:



titles=data["Title"].values
text=' '.join(titles)
titles


# In[ ]:


len(text)


# In[ ]:





# In[ ]:




