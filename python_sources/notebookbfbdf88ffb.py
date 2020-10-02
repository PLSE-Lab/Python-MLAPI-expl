#!/usr/bin/env python
# coding: utf-8

# first time to do kaggle 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')

# Import the 3 dimensionality reduction methods
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[ ]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


print (train.shape)


# In[ ]:


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(5.1,5.1))
plt.title('Correction plot of a 100 colums in the MNIST dataset')
# Draw the heatmap using seaborn
sns.heatmap(train.ix[:,100:200].astype(float).corr(),linewidths=0,square=True, cmap="cubehelix",xticklabels=False,yticklabels=False,annot=True)


# In[ ]:


# save the labels to a Pandas series target
target = train['label']
# Drop the label feature
train = train.drop("label",axis=1)


# In[ ]:


# Standardize the data
from sklearn.preprocessing import StandardScaler
X = train.values
X_std = StandardScaler().fit_transform(X)

# Caculating Eigenvetors and eigenvalues fo Cov matrix
mean_vec = np.mean(X_std, axis=0)
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# Create a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the eigenvalue, eigenvector pair from hight to low
eig_pairs.sort(key = lambda x:x[0], reverse = True)

# Calculation of Explained Variance from eigenvalues
tot = sum(eig_vals)
var_exp = [(i/tot) * 100 for i in sorted(eig_vals, reverse = True)] # Individual explained variance
cum_var_exp = np.cumsum(var_exp) # cumulative explained variance


# In[ ]:


trace1 = go.Scatter(
    x = list(range(784)),
    y = cum_var_exp,
    mode = 'lines+markers',
    name = 'Cumulative Explained Variance',
    hoverinfor = cum_var_exp,
    line = dict(
        shape = 'spline',
        color = 'goldenrod'
    )
)
trace2 = go.Scatter(
    x = list(range(784)),
    y = var_exp,
    mode = 'line+markers',
    name = 'Individual Explained Variance',
    hoverinfo = var_exp,
    line = dict(
        shape = 'linear',
        color = 'black'
    )
)

fig = tls.make_subplots(insets = [{'cell': (1,1),'1': 0.7, 'b': 0.5}],print_grid = True)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)
fig.layout.title = 'Explained Variance plot - Full and Zoomed-in'
fig.layout.xaxis = dict(range=[0, 80],title= 'Feature colums')
fig.layout.yaxis = dict(range=[0, 60],title='Explained Variace')
fig['data'] += [go.Scatter(x = list(range(784)), y= cum_var_exp, xaxis='x2',yaxis='y2',name = 'Cumulative Explained Variance')]
fig['data'] += [go.Scatter(x = list(range(784)),y=var_exp, xaxis='x2'),yaxis='y2',name='Individual Explained Variance']
py.plot(fig, file)

