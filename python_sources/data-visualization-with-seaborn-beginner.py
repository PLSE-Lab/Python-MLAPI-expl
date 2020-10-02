#!/usr/bin/env python
# coding: utf-8

# In this kernel, I try to visualize Breast Cancer Wisconsin (Diagnostic) Data Set. 
# I learn data visualization from  (https://www.kaggle.com/kanncaa1/seaborn-tutorial-for-beginners).
# Special thanks for him.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#Data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Reading data from file.
data=pd.read_csv('../input/data.csv',encoding="windows-1252")
#print(data)


# In[ ]:


col=data.columns
print(col)


# In[ ]:


# Data have 33 columns and 569 entries
data.info()


# In[ ]:


data.head(10)


# We use **diagnosis** as class label and drop **Unnamed: 32** from data.
# 
# 

# In[ ]:


# y includes our labels and x includes our features
y=data.diagnosis # M or B
list=['Unnamed: 32','id','diagnosis']
x=data.drop(list,axis=1)
x.head()


# In[ ]:


ax=sns.countplot(y,label="Count")   
B,M=y.value_counts()
print('Number of Benign: ', B)
print('Number of Malignant: ',M)


# In[ ]:


x.describe()


# Before visualization, we need to normalization or standardization processes. Because differences between values of features are very high to observe on plot. Features plotted in 3 groups and each group includes 10 features to observe better.

# In[ ]:





# In[ ]:


data_dia=y # data diagnosis
data=x     # dropped data  
data_n_2=(data-data.mean())/(data.std())
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",var_name="features",value_name='value') # melt can run without var_name and value_name
print(data)

#loc gets rows (or columns) with particular labels from the index. 
#iloc gets rows (or columns) at particular positions in the index (so it only takes integers).


# In[ ]:


# first ten features
data_dia=y # data diagnosis
data=x     # dropped data  
data_n_2=(data-data.mean())/(data.std())

# standardization
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",var_name="features",value_name='value')

# violin plot
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=90)


# In[ ]:


# box plot
plt.figure(figsize=(10,10))
sns.boxplot(x='features',y='value',hue='diagnosis',data=data)
plt.xticks(rotation=90)


# In[ ]:


# swarm plot
plt.figure(figsize=(10,10))
sns.swarmplot(x='features',y='value',hue='diagnosis',data=data)
plt.xticks(rotation=90)


# In[ ]:


# correlation map
f,ax = plt.subplots(figsize=(10, 10))
a=x.iloc[:,:10]
sns.heatmap(a.corr(), annot=True, linewidths=0.1,linecolor="red", fmt= '.1f',ax=ax)
plt.show()


# When we analyze plot, in  **radius_mean** feature, median of the Malignant and Benign looks like separated so it can be good for classification. However, in **fractal_dimension_mean** feature, median of Malignant and Benign does not looks like separated so it does not gives good information for classification.  

# In[ ]:


# Second ten features
data = pd.concat([y,data_n_2.iloc[:,10:20]],axis=1)
data = pd.melt(data, id_vars="diagnosis", var_name="features", value_name="value")

plt.figure(figsize=(10,10))
sns.violinplot(x="features",y='value',hue='diagnosis',data=data,split=True,inner="quart")
plt.xticks(rotation=90)
plt.show()


# In[ ]:


# boxplot
f,ax = plt.subplots(figsize=(10, 10))
sns.boxplot(x='features',y='value',hue='diagnosis',data=data)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


# swarm plot
plt.figure(figsize=(15,15))
sns.swarmplot(x='features',y='value',hue='diagnosis',data=data)
plt.xticks(rotation=90)


# In[ ]:


# correlation map
f,ax = plt.subplots(figsize=(10, 10))
a=x.iloc[:,10:20]
sns.heatmap(a.corr(), annot=True, linewidths=0.1,linecolor="red", fmt= '.1f',ax=ax)
plt.show()


# When we analyze plot, in  **perimeter_se** feature, median of the Malignant and Benign looks like separated so it can be good for classification. However, in **texture_se** feature, median of Malignant and Benign does not looks like separated so it does not gives good information for classification.  

# In[ ]:


# Rest of features
data = pd.concat([y,data_n_2.iloc[:,20:31]],axis=1)
data = pd.melt(data, id_vars="diagnosis", var_name="features", value_name="value")

plt.figure(figsize=(10,10))
sns.violinplot(x="features",y='value',hue='diagnosis',data=data,split=True,inner="quart")
plt.xticks(rotation=90)
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
sns.boxplot(x='features',y='value',hue='diagnosis',data=data)
plt.xticks(rotation=90)


# In[ ]:


# swarm plot
plt.figure(figsize=(15,15))
sns.swarmplot(x='features',y='value',hue='diagnosis',data=data)
plt.xticks(rotation=90)


# In[ ]:


# correlation map
f,ax = plt.subplots(figsize=(10, 10))
a=x.iloc[:,20:31]
sns.heatmap(a.corr(), annot=True, linewidths=0.1,linecolor="red", fmt= '.1f',ax=ax)
plt.show()


# When we analyze plot, in  **radius_worst**,**concavity_worst** feature, median of the Malignant and Benign looks like separated so it can be good for classification. 

# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[ ]:




