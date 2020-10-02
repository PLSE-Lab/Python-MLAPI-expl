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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt;
import os;
import seaborn as sns;


# In[ ]:


'''
    Goal: Analyse and visualize the dataset - bollywood-actors
'''
# Read the CSV file provided for bollywood actors;
df_actor = pd.read_csv('../input/bollywood-actors.csv');


# In[ ]:


# Get the shape of the dataframe
print(df_actor.shape);

'''
    The dataset contains 139 data points and 2 columns
'''

nRow, nCol = df_actor.shape;


# In[ ]:


# Have a look at the dataset

'''
    The dataset is already sorted in ascending order of Name column
'''

df_actor.head(2)


# In[ ]:


# Get the information about the dataset

'''
    The dataset includes two columns
        Name is object (contains the name of the actors)
        Height (in cm) (contains the height of the actors in centimeter)
'''
df_actor.info();


# In[ ]:


# Describe the dataset
df_actor.describe()


# In[ ]:


# Catch the null/ na percentage in the dataset

'''
    Dataset contains no null in any of the columns
    (which is boring!!)
'''

df_actor.isnull().sum() / nRow * 100


# In[ ]:


# Get the smallest actor in height
'''
    To get the min value you can use
    dataframe.nsmallest(n, columns)? But n is not known pre-assessment
'''

least_height = df_actor["Height(in cm)"].min();
small_actors = df_actor[df_actor["Height(in cm)"] == least_height];
print(small_actors);


# In[ ]:


# Get the tallest actor in height
'''
    Likewise you have to know the heighest value in height first
'''

max_height = df_actor["Height(in cm)"].max();
tall_actors = df_actor[df_actor["Height(in cm)"] == max_height];
print(tall_actors);


# In[ ]:


# Let's find whether we have any actors with same name?
if (df_actor.Name.unique().size == nRow):
    print("All actors are unique");
else:
    print("Found some!");


# In[ ]:


'''
    Let's find the variations the height
'''

df_actor["Height(in cm)"].value_counts().plot(kind='pie',
                                              figsize=(15,15),
                                              autopct='%1.0f%%',
                                              pctdistance=1.1, 
                                              labeldistance=1.2,
                                              title="Show the percentage of variations present in height");


# In[ ]:


'''
    Let's find the variations the height (this time count plot)
'''
from matplotlib import rcParams

# figure size in inches
rcParams['figure.figsize'] = 15,9
sns.countplot(df_actor["Height(in cm)"]);


# In[ ]:




