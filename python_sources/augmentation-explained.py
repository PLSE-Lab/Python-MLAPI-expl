#!/usr/bin/env python
# coding: utf-8

# # Data Augmentation
# 
# In this kernel I will show a way to augment the dataset so you will have twice as much training data. In order to do so we need to undertand how the original data was generated so first lets create some sandbox examples which will let us visualize the process of augmentation.
# 
# We are going to use [sklearn.datasets.make_classification ](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html) to generate data to play with.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.datasets import make_classification 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
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


# In[ ]:


cols = ['Real-life', 'Fantasy']
train, target = make_classification(10, 2, n_redundant=0, flip_y=0.0, n_informative=2, n_clusters_per_class=1, random_state=47, class_sep=1)
train = pd.DataFrame(train, columns=cols)
train['target'] = target
train


# In[ ]:


plt.figure(figsize=(14, 6))
sns.scatterplot(train[train['target'] == 0]['Real-life'], train[train['target'] == 0]['Fantasy'], s=150);
sns.scatterplot(train[train['target'] == 1]['Real-life'], train[train['target'] == 1]['Fantasy'], s=150);


# We have two separate cluster - one cluster per class, thus data labeled with 0's are on the left cluster and data labeled with 1's are on the right one.
# Next lets find centers for both of this clusters and visualize them.

# In[ ]:


plt.figure(figsize=(14, 6))
sns.scatterplot(train[train['target'] == 0]['Real-life'], train[train['target'] == 0]['Fantasy'], s=150);
sns.scatterplot(train[train['target'] == 1]['Real-life'], train[train['target'] == 1]['Fantasy'], s=150);
sns.scatterplot([train[train['target'] == 0][cols].mean().values[0]], [train[train['target'] == 0][cols].mean().values[1]], s=250);
sns.scatterplot([train[train['target'] == 1][cols].mean().values[0]], [train[train['target'] == 1][cols].mean().values[1]], s=250);


# Now lets take a closer look to the left cluster (the one labeled with 0's) and take a center point, which is called 'centroid', as an origin for our abscissa (X axis) and ordinate (Y axis).

# In[ ]:


plt.figure(figsize=(14, 6))
sns.scatterplot(train[train['target'] == 0]['Real-life'], train[train['target'] == 0]['Fantasy'], s=150);
sns.scatterplot([train[train['target'] == 0][cols].mean().values[0]], [train[train['target'] == 0][cols].mean().values[1]], s=250);
plt.plot([train[train['target'] == 0][cols].min()[0], train[train['target'] == 0][cols].max()[0]], [train[train['target'] == 0][cols].mean()[1]] * 2, sns.xkcd_rgb["black"]);
plt.plot([train[train['target'] == 0][cols].mean()[0]] * 2, [-1.25, 0.5], sns.xkcd_rgb["green"]);


# To augment our data we need to flip this points first around X axis (black line) and then around Y axis (green line). Let's do so.

# In[ ]:


plt.figure(figsize=(14, 6))
sns.scatterplot(train[train['target'] == 0]['Real-life'], train[train['target'] == 0]['Fantasy'], s=150);
sns.scatterplot([train[train['target'] == 0][cols].mean().values[0]], [train[train['target'] == 0][cols].mean().values[1]], s=250);
plt.plot([train[train['target'] == 0][cols].min()[0], train[train['target'] == 0][cols].max()[0]], [train[train['target'] == 0][cols].mean()[1]] * 2, sns.xkcd_rgb["black"]);
plt.plot([train[train['target'] == 0][cols].mean()[0]] * 2, [-1.25, 0.5], sns.xkcd_rgb["green"]);
sns.scatterplot(train[train['target'] == 0]['Real-life'].mean() + (train[train['target'] == 0]['Real-life'].mean() - train[train['target'] == 0]['Real-life']), train[train['target'] == 0]['Fantasy'], s=150);
mean = train[train['target'] == 0]['Real-life'].mean()
for x, y in train[train['target']==0][cols].values:
    x_new = mean + (mean - x)
    dx = x_new - x
    bias = 0.004
    if dx > 0:
        bias *= -1
    plt.arrow(x, y, dx + bias, 0, fc="k", ec="k", head_width=0.075, head_length=0.003, width=0.0025, color='green');


# Now we need to flip those new (green) points one more time, this time around Y axis. Looks a little messy, I know.

# In[ ]:


plt.figure(figsize=(14, 6))
sns.scatterplot(train[train['target'] == 0]['Real-life'], train[train['target'] == 0]['Fantasy'], s=150);
sns.scatterplot([train[train['target'] == 0][cols].mean().values[0]], [train[train['target'] == 0][cols].mean().values[1]], s=250);
plt.plot([train[train['target'] == 0][cols].min()[0], train[train['target'] == 0][cols].max()[0]], [train[train['target'] == 0][cols].mean()[1]] * 2, sns.xkcd_rgb["black"]);
plt.plot([train[train['target'] == 0][cols].mean()[0]] * 2, [-1.25, 0.5], sns.xkcd_rgb["green"]);
sns.scatterplot(train[train['target'] == 0]['Real-life'].mean() + (train[train['target'] == 0]['Real-life'].mean() - train[train['target'] == 0]['Real-life']), train[train['target'] == 0]['Fantasy'], s=150);
sns.scatterplot(train[train['target'] == 0]['Real-life'].mean() + (train[train['target'] == 0]['Real-life'].mean() - train[train['target'] == 0]['Real-life']), 
                train[train['target'] == 0]['Fantasy'].mean() + (train[train['target'] == 0]['Fantasy'].mean() - train[train['target'] == 0]['Fantasy']), s=150);
mean_x = train[train['target'] == 0]['Real-life'].mean()
mean_y = train[train['target'] == 0]['Fantasy'].mean()
for x, y in train[train['target']==0][cols].values:
    x_new = mean_x + (mean_x - x)
    y_new = mean_y + (mean_y - y)
    dx = x_new - x
    dy = y_new - y
    bias_x = 0.004
    bias_y = 0.1
    if dx > 0:
        bias_x *= -1
    if dy > 0:
        bias_y *= -1
    plt.arrow(x, y, dx + bias_x, 0, fc="k", ec="k", head_width=0.075, head_length=0.003, width=0.0025);
    plt.arrow(x_new, y, 0, dy + bias_y, fc="k", ec="k", head_width=0.002, head_length=0.075, width=0.0001);


# So lets cleenup a little and leave only original point and new, augmented ones. For every new point it's coordinates can be calculated using the following formula: Cluster_center * 2 - point_coordinates. So every point in the cluster will be flipped around cluster's 'origin point' (cluster's center).

# In[ ]:


plt.figure(figsize=(14, 6))
sns.scatterplot(train[train['target'] == 0]['Real-life'], train[train['target'] == 0]['Fantasy'], s=150);
sns.scatterplot([train[train['target'] == 0][cols].mean().values[0]], [train[train['target'] == 0][cols].mean().values[1]], s=250);
plt.plot([train[train['target'] == 0][cols].min()[0], train[train['target'] == 0][cols].max()[0]], [train[train['target'] == 0][cols].mean()[1]] * 2, sns.xkcd_rgb["black"]);
plt.plot([train[train['target'] == 0][cols].mean()[0]] * 2, [-1.25, 0.5], sns.xkcd_rgb["green"]);
augmented = 2 * train[train['target']==0][cols].mean() - train[train['target']==0][cols]
sns.scatterplot(augmented['Real-life'], augmented['Fantasy'], s=150);


# Now we have extended our cluster of points with new ones, which would be position no further from the center of the cluster than the maximum distance of the the most distant point to this center (I know this might be a difficult sentence to understand so spend some time and eventually this would make sencs to you).
# 
# This formula (Cluster_center * 2 - point) work in any dimensionality, but we are limited to only 3 dimensions with visualization, so let's plot 3d graph and see that it still works.
# 
# Here is an original cluster:

# In[ ]:


train, target = make_classification(100, 3, n_redundant=0, flip_y=0.0, n_informative=3, n_clusters_per_class=1, random_state=47, class_sep=3)
cols = ['Real-life', 'Fantasy', 'Landslide']
train = pd.DataFrame(train, columns=cols)
train['target'] = target

trace1 = go.Scatter3d(
    x=train[train['target'] == 0]['Real-life'],
    y=train[train['target'] == 0]['Fantasy'],
    z=train[train['target'] == 0]['Landslide'],
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
layout = go.Layout(
    margin=dict(l=0, r=0, b=0, t=0),
    scene = dict(
        xaxis = dict(
            title='Real-life'),
        yaxis = dict(
            title='Fantasy'),
        zaxis = dict(
            title='Landslide')
    )
)
fig = go.Figure(data=[trace1], layout=layout);
iplot(fig, filename='simple-3d-scatter', image_width=1024, image_height=768);


# And the same cluster with augmented points added:

# In[ ]:


augmented = 2 * train[train['target']==0][cols].mean() - train[train['target']==0][cols]
trace1 = go.Scatter3d(
    x=train[train['target'] == 0]['Real-life'],
    y=train[train['target'] == 0]['Fantasy'],
    z=train[train['target'] == 0]['Landslide'],
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
    x=augmented['Real-life'],
    y=augmented['Fantasy'],
    z=augmented['Landslide'],
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
            title='Real-life'),
        yaxis = dict(
            title='Fantasy'),
        zaxis = dict(
            title='Landslide')
    )
)
fig = go.Figure(data=[trace1, trace2], layout=layout);
iplot(fig, filename='simple-3d-scatter', image_width=1024, image_height=768);


# Ok, so far so good, but everything is while we are working with a single cluster. And this is possible because we have created our dataset using **n_clusters_per_class=1** parameter. But if you'd read make_classification's documentation you would see that by default it has values of **2**. In this case we need to find mean points (centroids) from both cluster. And if there are 3, 4, p clusters we need to find 2, 3, p centroids respectively. Otherwise an augmentation would go horribly wrong.
# 
# Let me show what I am talking about. Now we have almost the same dataset generated but with 2 clusters per each class. If we find only 1 single centroind and flip point around them they would be located in a wrong places.
# 
# First original points:

# In[ ]:


cols = ['Real-life', 'Fantasy']
train, target = make_classification(40, 2, n_redundant=0, flip_y=0.0, n_informative=2, n_clusters_per_class=2, random_state=47, class_sep=1)
train = pd.DataFrame(train, columns=cols)
train['target'] = target

plt.figure(figsize=(14, 8))
sns.scatterplot(train[train['target']==0]['Real-life'], train[train['target']==0]['Fantasy'], s=150);
sns.scatterplot([train[train['target']==0][cols].mean().values[0]], [train[train['target']==0][cols].mean().values[1]], s=250);


# Now adding an augmented ones

# In[ ]:


cols = ['Real-life', 'Fantasy']
train, target = make_classification(40, 2, n_redundant=0, flip_y=0.0, n_informative=2, n_clusters_per_class=2, random_state=47, class_sep=1)
train = pd.DataFrame(train, columns=cols)
train['target'] = target
augmented = 2 * train[train['target']==0][cols].mean() - train[train['target']==0][cols]

plt.figure(figsize=(14, 8))
sns.scatterplot(train[train['target']==0]['Real-life'], train[train['target']==0]['Fantasy'], s=150);
sns.scatterplot([train[train['target']==0][cols].mean().values[0]], [train[train['target']==0][cols].mean().values[1]], s=250);
sns.scatterplot(augmented['Real-life'], augmented['Fantasy'], s=150);


# Can you see the problem? No?

# By now we have been plotting only points with 0 labels. But lets add those with label 1:

# In[ ]:


cols = ['Real-life', 'Fantasy']
train, target = make_classification(40, 2, n_redundant=0, flip_y=0.0, n_informative=2, n_clusters_per_class=2, random_state=47, class_sep=1)
train = pd.DataFrame(train, columns=cols)
train['target'] = target
augmented = 2 * train[train['target']==0][cols].mean() - train[train['target']==0][cols]

plt.figure(figsize=(14, 8))
sns.scatterplot(train[train['target']==0]['Real-life'], train[train['target']==0]['Fantasy'], s=150);
sns.scatterplot([train[train['target']==0][cols].mean().values[0]], [train[train['target']==0][cols].mean().values[1]], s=250);
sns.scatterplot(augmented['Real-life'], augmented['Fantasy'], s=150);
sns.scatterplot(train[train['target']==1]['Real-life'], train[train['target']==1]['Fantasy'], s=150);


# Now we have mixed them up more than they were before adding an augmented data and any predictive model's performance would suffer because of this. 
# 
# Is there a way out? Yes there is. We need to find centroind for both of this clusters and make an augmnetation for them separately. In order to find centroind let me present you **K Means Clustering** algorithm. You can find an awesome explanation of this algorithm by Andrew Ng [over here](https://www.coursera.org/learn/machine-learning/lecture/93VPG/k-means-algorithm).
# 
# So lets use it and see how it will perform on our dataset.

# In[ ]:


km = KMeans(n_clusters=2)
km.fit(train[train['target']==0][cols]);
cenroinds = km.cluster_centers_
print('First cluster center:', cenroinds[0])
print('Second cluster center:', cenroinds[1])


# In[ ]:


plt.figure(figsize=(14, 8))
sns.scatterplot(train[train['target']==0]['Real-life'], train[train['target']==0]['Fantasy'], s=150);
sns.scatterplot([cenroinds[0][0]], [cenroinds[0][1]], s=250);
sns.scatterplot([cenroinds[1][0]], [cenroinds[1][1]], s=250);


# We have found a proper centroind for each cluster. Now it's time to make an augmentation for each of them separately.

# In[ ]:


clustered_df = train[train['target']==0][cols]
clustered_df['cluser'] = km.predict(train[train['target']==0][cols])
augmented_1 = 2 * clustered_df[clustered_df['cluser'] == 0][cols].mean() - clustered_df[clustered_df['cluser'] == 0][cols]
augmented_2 = 2 * clustered_df[clustered_df['cluser'] == 1][cols].mean() - clustered_df[clustered_df['cluser'] == 1][cols]

plt.figure(figsize=(14, 8))
sns.scatterplot(train[train['target']==0]['Real-life'], train[train['target']==0]['Fantasy'], s=150);
sns.scatterplot([cenroinds[0][0]], [cenroinds[0][1]], s=250);
sns.scatterplot([cenroinds[1][0]], [cenroinds[1][1]], s=250);
sns.scatterplot(augmented_1['Real-life'], augmented_1['Fantasy'], s=150);
sns.scatterplot(augmented_2['Real-life'], augmented_2['Fantasy'], s=150);


# It look much better. But now we are facing another problem - what if we don't know the number of clusters per class? Like we don't know them in this competition's data set. Well, we still can use KMean to try and find an optimal number of clusters. Let me show you how.

# In[ ]:


train, target = make_classification(1000, 4, n_redundant=0, flip_y=0.0, n_informative=3, n_clusters_per_class=3, random_state=47, class_sep=2)
train = pd.DataFrame(train)
train['target'] = target

inertia = dict()
for k in range(1, 6):
    km = KMeans(n_clusters=k)
    km.fit(train[train['target']==0][[0, 1, 2, 3]])
    inertia[k] = km.inertia_
    
plt.figure(figsize=(14, 8))
plt.plot(inertia.keys(), inertia.values());


# As you can see an inertia (a cummulative distance of all points to their cluster's centroid) is droping down significantly when we increase number of clusters from 1 to 2 and from 2 to 3 (which is a ground truth in this case), but almost does not decrease with 4 clusters and 5 clusters. So the best choise is 3 clusters. This is called an **elbow rule** (if you think about this graph as an arm than en elbow would be your best choice for number of clusters).

# Ok, so what does this all have to do with this competition's data? Well, since we have a [good guess](https://www.kaggle.com/mhviraf/synthetic-data-for-next-instant-gratification) that data was generated using make_classification all we need to do is to find a correct number of clusters per each class (we have a binary classification) per each of 512 data sets (we have 512 data sets, enumareted with **wheezy-copper-turtle-magic** feature. And then make an augmentation. 
# 
# Right now, if you assume that there is only 1 cluster per class and make an augmentation this only drops CV and LB. 
# 
# So maybe there are more than 1 cluster per class? Or maybe this dataset is to noisy so it is impossible to find a correct number of clusters? Looks like [Chris Deotte](https://www.kaggle.com/cdeotte) can get another golded kernel explaining this to us ;)
