#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Any results you write to the current directory are saved as output.


# In[ ]:


import os
print(os.listdir("../input/students-performance-in-exams"))


# In[ ]:


df = pd.read_csv("..//input//students-performance-in-exams//StudentsPerformance.csv")


# In[ ]:


df.head()


# In[ ]:


df.shape #Size of Data Frame


# In[ ]:


df.describe()


# In[ ]:


#Analysis the data correlation
corr=df.corr()
corr


# In[ ]:


#Checking for missing values
df.isnull().sum()


# In[ ]:


#Total Students By Gender
df['gender'].value_counts()


# In[ ]:


# Set theme
sns.set_style('whitegrid')


# In[ ]:


#Univariate Analysis for math score
sns.distplot(df['math score'],  bins=10);


# In[ ]:


#Univariate Analysis for reading score
sns.distplot(df['reading score'],  bins=10,color='green' );


# In[ ]:


#Univariate Analysis for writing score
sns.distplot(df['writing score'],  bins=10, color='purple' );


# In[ ]:


# Violin plot
sns.violinplot(x='gender', y='math score', data=df)


# In[ ]:


pkmn_type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]


# In[ ]:


# Set figure size with matplotlib
plt.figure(figsize=(10,6))
 
# Create plot
sns.violinplot(x='gender',
               y='math score', 
               data=df, 
               inner=None, # Remove the bars inside the violins
               palette=pkmn_type_colors)
 
sns.swarmplot(x='gender',
               y='math score', 
              data=df, 
              color='k', # Make points black
              alpha=0.7) # and slightly transparent
 
# Set title with matplotlib
plt.title('Math Score by Gender')


# In[ ]:


sns.heatmap(corr)


# In[ ]:


#  Bar Plot
sns.countplot(x='race/ethnicity', data=df, palette=pkmn_type_colors)
 
# Rotate x-labels
plt.xticks(rotation=-45)
plt.title('Data Analysis by Race/Ethnicity')


# In[ ]:


g = sns.FacetGrid(df, col="race/ethnicity", height=4, aspect=.5)
g.map(sns.barplot, "gender", "math score");


# In[ ]:


sns.catplot(x="gender", y="math score", hue="race/ethnicity", kind="bar", data=df);


# In[ ]:


sns.pairplot(df, hue="gender", height=2.5);


# In[ ]:


df_mean=df.groupby(
   ['gender'],as_index=True
).agg(
    {
         'math score':"mean",   
         'reading score': "mean", 
         'writing score': 'mean'  
    }
)
df_mean

