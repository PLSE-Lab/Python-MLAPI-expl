#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
import seaborn as sns
from sklearn.manifold import TSNE


# In[ ]:


data = pd.read_csv('../input/train.csv')
data.head()


# In[ ]:


label = data['label']
label.head()


# In[ ]:


data = data.drop('label',axis= 1)
data.head()


# In[ ]:


data.shape


# In[ ]:


label.shape


# In[ ]:


std_data = StandardScaler().fit_transform(data)
std_data.shape


# In[ ]:


std_data.T.shape


# In[ ]:


std_data.shape


# ###  t-SNE implementation with perplexity = 30, iteration=1000 (Default)

# In[ ]:


model = TSNE(n_components=2,random_state=0)


# In[ ]:


tsne_data = model.fit_transform(std_data)
tsne_data = np.vstack((tsne_data.T,label)).T
tsne_df = pd.DataFrame(data=tsne_data,columns=("f1","f2","label"))


# In[ ]:


sns.FacetGrid(tsne_df,hue="label",size=8).map(plt.scatter,'f1','f2').add_legend()
plt.show()


# In[ ]:





# > ###  t-SNE implementation with perplexity = 50, iteration=1000

# In[ ]:


model = TSNE(n_components=2,random_state=0,perplexity=50,n_iter=1000)


# In[ ]:


tsne_data = model.fit_transform(std_data)
tsne_data = np.vstack((tsne_data.T,label)).T
tsne_df = pd.DataFrame(data=tsne_data,columns=("f1","f2","label"))


# In[ ]:


sns.FacetGrid(tsne_df,hue="label",size=8).map(plt.scatter,'f1','f2').add_legend()
plt.show()


# ###  t-SNE implementation with perplexity = 100, iteration=2000

# In[ ]:


model = TSNE(n_components=2,random_state=0,perplexity=100,n_iter=2000)


# In[ ]:


tsne_data = model.fit_transform(std_data)
tsne_data = np.vstack((tsne_data.T,label)).T
tsne_df = pd.DataFrame(data=tsne_data,columns=("f1","f2","label"))


# In[ ]:


sns.FacetGrid(tsne_df,hue="label",size=8).map(plt.scatter,'f1','f2').add_legend()
plt.show()

