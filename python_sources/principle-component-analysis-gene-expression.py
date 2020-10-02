#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()


# # Analysis steps
# 
# 1. Remove columns that contain "Call" data
# 2. Transpose the dataframe so that each row is a patient and each column is a gene
# 3. Remove gene description header and set the gene accession numbers as the column headers
# 4. Split into train/test sets
# 5. Scale values to zero mean and unit varaince
# 6. PCA analysis
# 7. To do: K-means cluster

# In[ ]:


testfile = '../input/data_set_ALL_AML_independent.csv'
trainfile = '../input/data_set_ALL_AML_train.csv'
labels = '../input/actual.csv'

X_train = pd.read_csv(trainfile)
X_test = pd.read_csv(testfile)
y = pd.read_csv(labels)


# In[ ]:


# 1)  Remove "call" columns from training a test
train_keepers = [col for col in X_train.columns if "call" not in col]
test_keepers = [col for col in X_test.columns if "call" not in col]

X_train = X_train[train_keepers]
X_test = X_test[test_keepers]


# In[ ]:


# 2) Transpose
X_train = X_train.T
X_test = X_test.T
X_train.head()


# In[ ]:


# 3) Clean up the column names for training data
X_train.columns = X_train.iloc[1]
X_train = X_train.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)

# Clean up the column names for training data
X_test.columns = X_test.iloc[1]
X_test = X_test.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)

X_train.head()


# In[ ]:


# 4) Split into train and test 
X_train = X_train.reset_index(drop=True)
y_train = y[y.patient <= 38].reset_index(drop=True)

# Subet the rest for testing
X_test = X_test.reset_index(drop=True)
y_test = y[y.patient > 38].reset_index(drop=True)


# In[ ]:


# 5) Scale data
scaler = StandardScaler()
X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.fit_transform(X_test)
X_train[:3]


# # Principle Component Analysis
# 
# The analysis reveals that 21 principle components are needed to account for 80% of the variance. PC 1-3 add up to  about ~33% and the rest is a slow burn where each component after PC8 contributes between 1-2% of the variance up until PC38 which is essentially zero. 1% is a decent amonut of variance and so the number of important PCs is up for interpretation. 

# In[ ]:


# 6) PCA Analysis and projection
components = 21
pca = PCA(n_components=components)
Y = pca.fit(X_train_scl)
var_exp = Y.explained_variance_ratio_
cum_var_exp = np.cumsum(var_exp)


# In[ ]:


# Plot the explained variance
x = ["PC%s" %i for i in range(1,components)]
trace1 = go.Bar(
    x=x,
    y=list(var_exp),
    name="Explained Variance")

trace2 = go.Scatter(
    x=x,
    y=cum_var_exp,
    name="Cumulative Variance")

layout = go.Layout(
    title='Explained variance',
    xaxis=dict(title='Principle Components', tickmode='linear'))

data = [trace1, trace2]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig)


# # Projection of first three components
# The first three components only explain 33% of the variance but we'll go ahead plot the projection to get a visual of it. 

# In[ ]:


# Project first three components
Y_train_pca = pca.fit_transform(X_train_scl)

traces = []
for name in ['ALL', 'AML']:
    trace = go.Scatter3d(
        x=Y_train_pca[y_train.cancer==name,0],
        y=Y_train_pca[y_train.cancer==name,1],
        z=Y_train_pca[y_train.cancer==name,2],
        mode='markers',
        name=name,
        marker=go.Marker(size=10, line=go.Line(width=1),opacity=1))
    
    traces.append(trace)

layout = go.Layout(
    xaxis=dict(title='PC1'),
    yaxis=dict(title='PC2'),
    title="Projection of First Three Principle Components"
)

data = traces
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig)

