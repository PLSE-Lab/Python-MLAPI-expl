#!/usr/bin/env python
# coding: utf-8

# In this project ,we will try to predict the forest cover type ,inorder to achieve this goal we first need to explore and imporve each section in the database using pandas for data proccessing and matplotlib for data anlysis on graphs.

# We start by importing the libraries we will use.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
from collections import Counter
import matplotlib.pyplot as plt
#import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()

from sklearn.decomposition import PCA
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
#from scipy.stats import linregress
import scipy
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.import zipfilepath.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# importing the database csv file and getting a general overview of the dataset.

# In[ ]:


import zipfile

zf=zipfile.ZipFile('/kaggle/input/forest-cover-type-kernels-only/train.csv.zip')
zf2=zipfile.ZipFile('/kaggle/input/forest-cover-type-kernels-only/test.csv.zip')

sample_submission= pd.read_csv('../input/forest-cover-type-kernels-only/sample_submission.csv.zip')
sampleSubmission= pd.read_csv('../input/forest-cover-type-kernels-only/sampleSubmission.csv.zip')
train= pd.read_csv(zf.open('train.csv'))
test= pd.read_csv(zf2.open('test.csv'))

print('train (r,c)=',train.shape)
print('test (r,c)=',test.shape)
print('sample_submission (r,c)=',sample_submission.shape)
print('sampleSubmission (r,c)=',sampleSubmission.shape)


# Overview of train dataset

# In[ ]:


train.head()


# In[ ]:


test.head().describe()


# Overview of test dataset

# In[ ]:


test.head()


# In[ ]:


test.head().describe()

Overview of sample_submission dataset
# In[ ]:


sample_submission.head()


# In[ ]:


sample_submission.head().describe()


# Overview of sampleSubmission dataset

# In[ ]:


sampleSubmission.head()


# In[ ]:


sampleSubmission.head().describe()


# > Now based on our observation of the dataset we will change its indexing to be based on the ID .

# In[ ]:


train.set_index('Id')


# In[ ]:


test.set_index('Id')


# In[ ]:


sample_submission.set_index('Id')


# In[ ]:


sampleSubmission.set_index('Id')


# The next step is to identify the data values with Nan values in the train dataset that we will work with and replace them with 'unkown'.

# In[ ]:


print('Number of null values in the dataset=', len(train[train.isnull()]) )

Replacing the null values
# In[ ]:


train.fillna('unknown')


# We will make a simple graph to see the realation between Elevation and Aspect by making a perfect fit line in each graph.
# By this perfect fit line we will be able to judge the realtionship between Elevation and Aspect.

# In[ ]:


plt.plot(train.Elevation, train.Aspect)
plt.show()

# plt.show()


# As we can see its hard to judge the realationship between them by ploting them as raw data so we will only use thier mean,number,min and max

# In[ ]:


#first we group the data by its elevation and compare it by the number of aspects , minumin of aspects and max of aspects
p=train.groupby(['Elevation']).Aspect.agg([len, min, max])

#plt.scatter(p)
plt.plot(p.len)
plt.xlabel('Elevation')
plt.ylabel('Number of Apects')
plt.show()  # or plt.savefig("name.png")
# plt.scatter(x, y)
# y = train.
# z=
# plt.scatter(train.Elevation, )
# plt.show()  # or plt.savefig("name.png")


# In[ ]:


p=train.groupby(['Elevation']).Aspect.mean()
p.astype('int')
plt.plot(p)
plt.show()


# In[ ]:


x=train.Elevation.mean()
plt.scatter(x,train.Aspect.mean())

plt.show()


# Now to the main task which is clarifying each tree cover type according to its ID

# In[ ]:


#row1 is Id OF train row2 is Id of Cover_Type
# def add(row1,row2) :
#     if (row1.Id == row2.Id):
#         x= pd.concat([row1, row2.Cover_Type])
#     else:
#         print('unkown value')
#     return x    
# 


# In[ ]:


left= train.set_index('Id')
right= sample_submission.set_index('Id')
output=left.join(right, lsuffix='_')
final_data = output.drop(columns="Cover_Type")
final_data


# Submiting the final output

# In[ ]:


final_data.to_csv('FinalSampleSubmition', index = False)

