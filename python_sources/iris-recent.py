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


import pandas as pd 
import numpy as np
import seaborn as sns
df = pd.read_csv('/kaggle/input/iris/Iris.csv')
df


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df1 = df.drop("Id",axis=1)


# In[ ]:


df1


# In[ ]:


# How many data points are there in this dataset
df1.shape


# In[ ]:


# What are the  columns in dataset
df1.columns


# In[ ]:


# How many points are there for each class.
print(df1["Species"].value_counts())
# Balanced vs Imbalanced Dataset
# Iris dataset is an example of balanced dataset


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
df1.plot(kind="scatter",x="SepalWidthCm",y="SepalLengthCm")


# In[ ]:


# 

import seaborn as sns
x = sns.FacetGrid(df1,hue="Species",size=5)
x = x.map(plt.scatter,x='SepalWidthCm',y='PetalWidthCm')
x = x.add_legend()
plt.show()


# In[ ]:


sns.barplot(x='PetalWidthCm',y="Species",data=df1)


# In[ ]:


sns.barplot(x="Species",y='PetalWidthCm',data=df1,orient="vertical",estimator=np.median)


# In[ ]:


sns.barplot(x="Species",y='PetalWidthCm',data=df1,orient="vertical",estimator=np.mean)


# In[ ]:


sns.stripplot(x="Species",y='PetalWidthCm',data=df1,orient="vertical",jitter=0.1,edgecolor="grey",linewidth=1,dodge=True,marker="*")


# In[ ]:


sns.swarmplot(x="Species",y='PetalWidthCm',data=df1,orient="vertical")


# In[ ]:


# Pair plot used to visualize graphs with more than 2 dim
sns.pairplot(df1,hue="Species").map_diag(plt.hist).add_legend()


# In[ ]:


# Pair plot used to visualize graphs with more than 2 dim
sns.pairplot(df1,hue="Species").map_diag(sns.distplot).add_legend()


# In[ ]:


sns.pairplot(df1,hue="Species").map_diag(sns.swarmplot).add_legend()


# In[ ]:


g = sns.pairplot(df1,hue="Species")
g = g.map_diag(plt.hist)
g = g.map_offdiag(sns.swarmplot)
g = g.add_legend();
#g.set_xticklabels(rotation=270);


# In[ ]:


x = sns.FacetGrid(df,hue="Species",size=10)
x = x.map(sns.swarmplot,"SepalWidthCm","PetalWidthCm")
x = x.add_legend()


# In[ ]:


# Barplot
plt.figure(figsize=(10,10))
sns.barplot(x="Species",y="SepalWidthCm",data=df,ci=0,palette="spring_r",saturation=0)


# In[ ]:


# Distplot
sns.distplot(df1["SepalLengthCm"],hist=False);


# In[ ]:


# Distplot
sns.distplot(df1["SepalLengthCm"],kde=False);


# In[ ]:


# Distplot

sns.catplot(data=df1,kind="bar",x="SepalLengthCm",y="Species",ci=0)
#x = x.map(kind="kde",x="SepalLengthCm")


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(30,30));
sns.set_style("darkgrid");
g = sns.catplot(x="SepalLengthCm", y="SepalWidthCm", hue="Species", kind="point", data=df1);
g.set_xticklabels(rotation=270);


# In[ ]:


sns.countplot(data=df,x="Species");


# In[ ]:


sns.catplot(x="Species",kind="count",palette="spring",data=df);


# In[ ]:


sns.jointplot(data=df,hue="Species",kind="kde",x="SepalLengthCm",y="SepalWidthCm");


# In[ ]:


sns.jointplot(data=df,hue="Species",kind="kde",x="SepalLengthCm",y="SepalWidthCm",color="r");


# In[ ]:


plt.scatter("SepalLengthCm","PetalLengthCm",data=df);


# In[ ]:


#!python -m pip install -U matplotlib


# In[ ]:


get_ipython().system('pip install matplotlib')


# In[ ]:


# Randomly sample data points
import random
random.random()


# ** Above function will generate a new random variable in thee range 0.0 to 1.0**

# In[ ]:


df1


# In[ ]:


d = df1.values
d


# In[ ]:


n = 150 # Total data points
m = 100 # No of elements in sample
p = m/n; # Probability of getting 30 points from 150 data points


sample_data = []
for i in range(0,n):
    a = random.random()
    print(a)
    if a<=p:
        sample_data.append(d[i,:])


# In[ ]:


sample_data


# In[ ]:


sample_data_df = pd.DataFrame(sample_data,columns= ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm','Species'])


# In[ ]:


sample_data_df


# In[ ]:


sns.barplot(x = "Species",y="SepalLengthCm",data=sample_data_df);


# In[ ]:


plt.figure(figsize=(10,8))
sns.violinplot(x = "Species",y="SepalLengthCm",data=sample_data_df);


# In[ ]:


plt.figure(figsize=(10,8))
sns.boxplot(x = "Species",y="SepalLengthCm",data=sample_data_df);


# In[ ]:


plt.figure(figsize=(10,8))
sns.violinplot(x = "Species",y="SepalLengthCm",data=sample_data_df,order = sorted(df.Species.unique()));


# In[ ]:


# plt.figure(figsize=(10,8))
# sns.violinplot(x = "Species",y="SepalLengthCm",data=sample_data_df,order = sorted(df.Species.unique()),orient="h");
# plt.show()


# In[ ]:


plt.figure(figsize=(10,8))
sns.violinplot(x = "Species",y="SepalLengthCm",data=sample_data_df,order = sorted(df.Species.unique()),fliersize=20);


# In[ ]:


plt.figure(figsize=(10,8))
sns.violinplot(x = "Species",y="SepalLengthCm",data=sample_data_df,order = sorted(df.Species.unique()),bw=0.05);


# In[ ]:


plt.figure(figsize=(10,8))
sns.violinplot(x = "Species",y="SepalLengthCm",data=sample_data_df,order = sorted(df.Species.unique()),bw=5);


# In[ ]:


plt.figure(figsize=(10,8))
sns.violinplot(x = "Species",y="SepalLengthCm",data=sample_data_df,order = sorted(df.Species.unique()),bw=0.2);


# In[ ]:


plt.figure(figsize=(10,8))
sns.violinplot(x = "Species",y="SepalLengthCm",data=sample_data_df,order = sorted(df.Species.unique()),bw=0.15,cut=2,scale="area",inner="box",dodge=True);


# In[ ]:


# import seaborn as sns
# sns.get_dataset_names()


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'sns.get_dataset_names')


# In[ ]:


# p.get_figure().savefig('../../figures/violinplot.png')


# In[ ]:





# In[ ]:




