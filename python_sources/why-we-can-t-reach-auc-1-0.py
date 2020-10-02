#!/usr/bin/env python
# coding: utf-8

# # Preamble
# 
# After [mhviraf's](https://www.kaggle.com/mhviraf) good guess of [how data for this competition was generated](https://www.kaggle.com/mhviraf/synthetic-data-for-next-instant-gratification) I have played with [sklearn.datasets.make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html) for a while.
# 
# And the main reason why it is impossible (well, almost impossible) to reach AUC of 1.0 is it's parameter **flip_y**.
# Here is it's description:
# 
# > flip_y : float, optional (default=0.01)
# >
# > The fraction of samples whose class are randomly exchanged. 
# >
# > Larger values introduce noise in the labels and make the classification task harder.
# 
# The default values is 0.01, which means that 1% of data would be randomly flipped from 0 to 1 or vice versa.
# 
# It means that if sample was originaly generated to be in the cluster of 0's it's target would be changed to 1 and any model would missclassify it.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.datasets import make_classification 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.io as pio
from plotly.offline import init_notebook_mode, iplot
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
init_notebook_mode()
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# At first let's generate a 'perfect' dataset with 1000 samples and only 2 features so we can easily display it. Both of this features will be informative (n_informative=2). 
# 
# You can find out what is the difference between informative features and noise in [this kernel](https://www.kaggle.com/cdeotte/support-vector-machine-0-925).
# 
# Also for the sake of visibility parameter **class_sep** will be set to 3, so two clusters would be far away from each other.

# In[ ]:


train, target = make_classification(1000, 2, n_redundant=0, flip_y=0.0, n_informative=2, n_clusters_per_class=1, random_state=47, class_sep=3)
cols = ['Bohemian-rhapsody', 'Dont-stop-me-now']
train = pd.DataFrame(train, columns=cols)
train['target'] = target


# ## 2 Dimensions

# In[ ]:


plt.figure(figsize=(14, 8))
sns.scatterplot(train[train['target'] == 0]['Bohemian-rhapsody'], train[train['target'] == 0]['Dont-stop-me-now']);
sns.scatterplot(train[train['target'] == 1]['Bohemian-rhapsody'], train[train['target'] == 1]['Dont-stop-me-now']);


# ## 3 Dimensions

# In[ ]:


train, target = make_classification(500, 3, n_redundant=0, flip_y=0.0, n_informative=3, n_clusters_per_class=1, random_state=47, class_sep=3)
cols = ['Bohemian-rhapsody', 'Dont-stop-me-now', 'Killer-Queen']
train = pd.DataFrame(train, columns=cols)
train['target'] = target

trace1 = go.Scatter3d(
    x=train[train['target'] == 0]['Bohemian-rhapsody'],
    y=train[train['target'] == 0]['Dont-stop-me-now'],
    z=train[train['target'] == 0]['Killer-Queen'],
    mode='markers',
    marker=dict(
        size=8,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=1
    )
)
trace2 = go.Scatter3d(
    x=train[train['target'] == 1]['Bohemian-rhapsody'],
    y=train[train['target'] == 1]['Dont-stop-me-now'],
    z=train[train['target'] == 1]['Killer-Queen'],
    mode='markers',
    marker=dict(
        size=8,
        line=dict(
            color='rgba(100, 100, 100, 0.14)',
            width=0.5
        ),
        opacity=1
    )
)
layout = go.Layout(
    margin=dict(l=0, r=0, b=0, t=0),
    scene = dict(
        xaxis = dict(
            title='Bohemian-Rhapsody'),
        yaxis = dict(
            title='Dont-stop-me-now'),
        zaxis = dict(
            title='Killer-Queen')
    )
)
fig = go.Figure(data=[trace1, trace2], layout=layout);
iplot(fig, filename='simple-3d-scatter', image_width=1024, image_height=768);


# It is easy to separate this two clusters and almost any model would gain an AUC of 1.0.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train[cols], train['target'], test_size=0.2, random_state=47)
clf = QuadraticDiscriminantAnalysis(reg_param=0.6)
clf.fit(X_train, y_train)
print('ROC AUC:', roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))


# # Introducing flip_y
# Now it's time to flip some of the the labels. 
# 
# Parameter **flip_y** is going to be set to 0.1, so ~10% of the data would be misslabeled.

# In[ ]:


train, target = make_classification(1000, 2, n_redundant=0, flip_y=0.1, n_informative=2, n_clusters_per_class=1, random_state=47, class_sep=3)
cols = ['Bohemian-rhapsody', 'Dont-stop-me-now']
train = pd.DataFrame(train, columns=cols)
train['target'] = target


# ## 2 Dimensions

# In[ ]:


plt.figure(figsize=(14, 8))
sns.scatterplot(train[train['target'] == 0]['Bohemian-rhapsody'], train[train['target'] == 0]['Dont-stop-me-now']);
sns.scatterplot(train[train['target'] == 1]['Bohemian-rhapsody'], train[train['target'] == 1]['Dont-stop-me-now']);


# ## 3 Dimensions

# In[ ]:


train, target = make_classification(500, 3, n_redundant=0, flip_y=0.1, n_informative=3, n_clusters_per_class=1, random_state=47, class_sep=3)
cols = ['Bohemian-rhapsody', 'Dont-stop-me-now', 'Killer-Queen']
train = pd.DataFrame(train, columns=cols)
train['target'] = target

trace1 = go.Scatter3d(
    x=train[train['target'] == 0]['Bohemian-rhapsody'],
    y=train[train['target'] == 0]['Dont-stop-me-now'],
    z=train[train['target'] == 0]['Killer-Queen'],
    mode='markers',
    marker=dict(
        size=8,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=1
    )
)
trace2 = go.Scatter3d(
    x=train[train['target'] == 1]['Bohemian-rhapsody'],
    y=train[train['target'] == 1]['Dont-stop-me-now'],
    z=train[train['target'] == 1]['Killer-Queen'],
    mode='markers',
    marker=dict(
        size=8,
        line=dict(
            color='rgba(100, 100, 100, 0.14)',
            width=0.5
        ),
        opacity=1
    )
)
layout = go.Layout(
    margin=dict(l=0, r=0, b=0, t=0),
    scene = dict(
        xaxis = dict(
            title='Bohemian-Rhapsody'),
        yaxis = dict(
            title='Dont-stop-me-now'),
        zaxis = dict(
            title='Killer-Queen')
    )
)
fig = go.Figure(data=[trace1, trace2], layout=layout);
iplot(fig, filename='simple-3d-scatter', image_width=1024, image_height=768);


# Now we can see that there are two clusters of points with some of them 'flipped' and any model would have a missclassifications.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train[cols], train['target'], test_size=0.2, random_state=47)
clf = QuadraticDiscriminantAnalysis(reg_param=0.6)
clf.fit(X_train, y_train)
print('ROC AUC:', roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))


# And at the end - something that mimic a dataset provided in this competition.

# In[ ]:


train, target = make_classification(1000, 3, n_redundant=0, flip_y=0.08, n_informative=2, n_clusters_per_class=2, random_state=47, class_sep=1)
cols = ['Bohemian-rhapsody', 'Dont-stop-me-now', 'Killer-Queen']
train = pd.DataFrame(train, columns=cols)
train['target'] = target

trace1 = go.Scatter3d(
    x=train[train['target'] == 0]['Bohemian-rhapsody'],
    y=train[train['target'] == 0]['Dont-stop-me-now'],
    z=train[train['target'] == 0]['Killer-Queen'],
    mode='markers',
    marker=dict(
        size=8,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=1
    )
)
trace2 = go.Scatter3d(
    x=train[train['target'] == 1]['Bohemian-rhapsody'],
    y=train[train['target'] == 1]['Dont-stop-me-now'],
    z=train[train['target'] == 1]['Killer-Queen'],
    mode='markers',
    marker=dict(
        size=8,
        line=dict(
            color='rgba(100, 100, 100, 0.14)',
            width=0.5
        ),
        opacity=1
    )
)
layout = go.Layout(
    margin=dict(l=0, r=0, b=0, t=0),
    scene = dict(
        xaxis = dict(
            title='Bohemian-Rhapsody'),
        yaxis = dict(
            title='Dont-stop-me-now'),
        zaxis = dict(
            title='Killer-Queen')
    )
)
fig = go.Figure(data=[trace1, trace2], layout=layout);
iplot(fig, filename='simple-3d-scatter', image_width=1024, image_height=768);


# In[ ]:


fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(18, 8))
sns.kdeplot(train[train['target'] == 0]['Bohemian-rhapsody'], ax=axes[0]);
sns.kdeplot(train[train['target'] == 1]['Bohemian-rhapsody'], ax=axes[0]);
sns.kdeplot(train[train['target'] == 0]['Dont-stop-me-now'], ax=axes[1]);
sns.kdeplot(train[train['target'] == 1]['Dont-stop-me-now'], ax=axes[1]);
sns.kdeplot(train[train['target'] == 0]['Killer-Queen'], ax=axes[2]);
sns.kdeplot(train[train['target'] == 1]['Killer-Queen'], ax=axes[2]);

