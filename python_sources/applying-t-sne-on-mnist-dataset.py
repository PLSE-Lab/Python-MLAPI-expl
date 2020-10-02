#!/usr/bin/env python
# coding: utf-8

# ### IMPORTING REQUIRED LIBRARIES[](http://)

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# importing dataset
df = pd.read_csv('../input/train.csv')

# features
train_x = df.drop('label', axis = 1)
print(train_x.head()) # first five rows

#label
train_y = df['label']
print(train_y.head()) # first five rows


# In[3]:


# shape of our data: (rows, columns)
print('train_x shape:', train_x.shape)
print('train_y.shape:', train_y.shape)


# In[4]:


# standardization of data
sc = StandardScaler()
train_x_scaled = sc.fit_transform(train_x)
print('Shape of train_x_scaled: ', train_x_scaled.shape)


# In[5]:


t_sne = TSNE(random_state = 0)
t_sne.n_components = 2 # no. of features we want after dimensionality reduction
t_sne.perplexity = 50 # perplexity is related to the number of nearest neighbors
# learning rate is default to 200
print(t_sne)
tsne_data = t_sne.fit_transform(train_x_scaled[0: 5000,:])
print(tsne_data)


# In[6]:


tsne_data = np.vstack((tsne_data.T, train_y[0:5000])).T
tsne_df = pd.DataFrame(tsne_data, columns = ['Feature1', 'Feature2', 'Label'])
print(tsne_df.head())


# In[7]:


sns.FacetGrid(tsne_df, hue = 'Label', height = 6) .map(plt.scatter, 'Feature1', 'Feature2') .add_legend()
plt.title('perplexity = 50   n_iter = 1000')
plt.show()


# TUNING T-SNE PARAMETERS AND VISUALIZING

# In[8]:


t_sne.perplexity = 50 # same as before
t_sne.n_iter = 2000 # before it was default i.e 1000
print(t_sne)
t_sne_data2 = t_sne.fit_transform(train_x_scaled[0:5000,:])

t_sne_data2 = np.vstack((t_sne_data2.T, train_y[:5000])).T
t_sne_df2 = pd.DataFrame(t_sne_data2, columns = ['Feature1', 'Feature2', 'Label'])
sns.FacetGrid(t_sne_df2, hue = 'Label', height = 6) .map(plt.scatter, 'Feature1', 'Feature2') .add_legend()
plt.title('perplexity = 50   n_iter = 2000')
plt.show()


# In[9]:


t_sne.perplexity = 20 # before it was 50
t_sne.n_iter = 500 # before it was 2000
print(t_sne)
t_sne_data3 = t_sne.fit_transform(train_x_scaled[0:5000,:])

t_sne_data3 = np.vstack((t_sne_data3.T, train_y[:5000])).T
t_sne_df3 = pd.DataFrame(t_sne_data3, columns = ['Feature1', 'Feature2', 'Label'])
sns.FacetGrid(t_sne_df3, hue = 'Label', height = 6) .map(plt.scatter, 'Feature1', 'Feature2') .add_legend()
plt.title('perplexity = 20   n_iter = 500')
plt.show()


# In[10]:


t_sne.perplexity = 100 # before it was 20
t_sne.n_iter = 5000 # before it was 500
print(t_sne)
t_sne_data4 = t_sne.fit_transform(train_x_scaled[0:5000,:])

t_sne_data4 = np.vstack((t_sne_data4.T, train_y[:5000])).T
t_sne_df4 = pd.DataFrame(t_sne_data4, columns = ['Feature1', 'Feature2', 'Label'])
sns.FacetGrid(t_sne_df4, hue = 'Label', height = 6) .map(plt.scatter, 'Feature1', 'Feature2') .add_legend()
plt.title('perplexity = 100   n_iter = 5000')
plt.show()


# In[11]:


t_sne.perplexity = 5000 # perplexity equals to nummber of point, lets see
t_sne.n_iter = 2000 # before it was 5000
print(t_sne)
t_sne_data5 = t_sne.fit_transform(train_x_scaled[0:5000,:])

t_sne_data5 = np.vstack((t_sne_data5.T, train_y[:5000])).T
t_sne_df5 = pd.DataFrame(t_sne_data5, columns = ['Feature1', 'Feature2', 'Label'])
sns.FacetGrid(t_sne_df5, hue = 'Label', height = 6) .map(plt.scatter, 'Feature1', 'Feature2') .add_legend()
plt.title('perplexity = 5000  n_iter = 2000')
plt.show()


# In[12]:


t_sne.perplexity = 2 # very small perplexity, lets see
t_sne.n_iter = 2000 # before it was default i.e 5000
print(t_sne)
t_sne_data6 = t_sne.fit_transform(train_x_scaled[0:5000,:])

t_sne_data6 = np.vstack((t_sne_data6.T, train_y[:5000])).T
t_sne_df6 = pd.DataFrame(t_sne_data6, columns = ['Feature1', 'Feature2', 'Label'])
sns.FacetGrid(t_sne_df6, hue = 'Label', height = 6) .map(plt.scatter, 'Feature1', 'Feature2') .add_legend()
plt.title('perplexity = 2  n_iter = 2000')
plt.show()


# we should always try to run t-sne with increasingly different iterations and perplexity until the plot stabilizes and starts giving almost same results

# #### REFERENCES :
# 1. Applied AI Course: [ https://www.appliedaicourse.com ]
# 2. How to use T-SNE Effectively : [ https://distill.pub/2016/misread-tsne/ ]
