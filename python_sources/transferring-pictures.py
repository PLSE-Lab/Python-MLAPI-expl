#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Let's look a general summary of our data and data types
# 

# In[ ]:


df = pd.read_csv('../input/akosombo.csv')
df.describe()


# We can see that we have 728 images, however only 696 have unique file names. This is probably because the file names were tampered with during the collection process. We are not seeing the mean and other descriptive factors. Let's look at the data types

# In[ ]:


df.dtypes


# We are not seeing other descriptive factor because all our columns are objects. Let's check whether we have any null values and then convert the size, speed and eta to float

# In[ ]:


pd.isnull(df).any()


# We dont have any null values. If we did we would have to drop them or fill them depending on the situation. We now do a few things 
# 1. Change eta to be in seconds
# 2. Remove % sign from the percentage snd change the data type
# 2. Remove the KBs from the size
# 3. Convert the MB/s for speed to KB/s to be able to work with the same units and the drop the units 
# 

# In[ ]:


#Removing the hours part of the eta.
df['eta'] = df.eta.map(lambda x: x.split(':')[-1])
df['percentage'] = df['percentage'].apply(lambda x: x.split('%')[0])
df['percentage'] = df['percentage'].astype(float)
df['size'] = df['size'].map(lambda x: x.split('KB')[0])
df['size'] = df['size'].astype(float)


# In[ ]:


speed = df['speed']
for sp in speed:
    ext = sp[-4:]
    if ext == 'MB/s':
        sp = sp.split('MB/s')[0] 
        sp= float(sp)
        sp = sp*1024 
    elif ext =='KB/s':
        sp = sp.split('KB/s')[0]
        sp=float(sp)
    df['speed'] = sp


# In[ ]:


#I am sure the baove code can be written in a better way. I am open to suggestions
df.head()


# In[ ]:


df['eta'] = df['eta'].astype(float)
df.describe()


# The average rate of transfer is 1.12 seconds. The maximun eta is 4 secs and the minimu is 0 seconds meaning the file transfer took miliseconds. The largest file is 9010 KBs. How long did it take to transfer the file?

# In[ ]:


df[df['size']==9010]


# It took just 3 secs to transfer the largest file yet the maximum taken for all file transfers was 4 secs. It probably means that speed for the files that took 4 secs was slower. Let's find out

# In[ ]:


df[df['eta']==4]


# The images were much less size compared to the largest file but took more time to transfer because the speed od transfer was much slower. Is there any relationship between the size of a file and the speed.

# In[ ]:


sns.jointplot(data=df, x='speed',y='size')


# There is no relationship between the file size and speed of transfer. I sthere a relationship between the size of an image and the eta?

# In[ ]:


sns.jointplot(data=df, x='eta',y='size')


# There is a very strong positive relationship between the size of an image and how long it takes to trnasfer to the other computer. The larger the image the longer it will take to transfer. 

# #How long did it take to transfer all files 
# 

# In[ ]:


total = df['eta'].sum()
mins = total/60
mins


# Looking at the distribution of the eta

# In[ ]:



sns.distplot(df['eta'],bins=10)


# In[ ]:


#Size distribution 
sns.distplot(df['size'],bins=30)


# This is definately a very basic dataset, so will stop it here
