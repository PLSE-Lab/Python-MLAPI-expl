#!/usr/bin/env python
# coding: utf-8

# # Introduction
# In our training data we have more than 200 features with different charachteristics like type, dirstribution, range, outliers, level of sparsity among others.  It's hard and time consuming to deep anlyze them one by one, and then proccess them differently before we use them in our model (Apples $\neq$ Oranges).  Let's do part of this work here.
# 
# Let's use some features from the `application_train.csv` data to maximize the common good since most of us use it extensively in modeling.  The same analysis applies to other numerical features.
# 
# The customer's total income (`AMT_INCOME_TOTAL`) and credit amount (`AMT_CREDIT`) are among the factors used by credit bureaus in credit scoring.  A credit score is considered a good predictor of the borrower's ability to pay back their dept on time.
# 
# The income amount, however, have a few marginal outlier values that could slow down or even prevent convergence of some gradient based algorithms.  Moreover, the distribution of the values is heavily skewed for both features, and they have different range values compared to the other features.  The negative effect of having different feature ranges is less pronounced in tree based algorithms, though.
# 
# In the two figures below, on the left we have the customer income plotted against an arbitrary customer number to show marginal outlier values.  On the right plot we see big skewness in the distribution of the credit amount of the loan.
# 
# [1]: http://www.ccs.neu.edu/home/vip/teach/MLcourse/4_boosting/lecture_notes/boosting/boosting.pdf

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os

apps = pd.read_csv('../input/application_train.csv', index_col='SK_ID_CURR')
target = apps.pop('TARGET')


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

X = apps[['AMT_CREDIT', 'AMT_INCOME_TOTAL']]

ax = axes[0]
income = X['AMT_INCOME_TOTAL']
ax.scatter(np.arange(income.shape[0]), income)
ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.set_title('Marginal outliers in customer income feature')
ax.set_xlabel('Customer number (arbitrary)')
ax.set_ylabel('Customer income')

ax = axes[1]
credit = X['AMT_CREDIT']
sns.distplot(credit, ax=ax, kde=False)
plt.xticks(rotation=30)
ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.set_title('Highly skewed distribution for credit amount of loan')
ax.set_xlabel('Credit amount')
ax.set_ylabel('Number of customers');


# In[ ]:


# Author:  Raghav RV <rvraghav93@gmail.com>
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Thomas Unterthiner
# License: BSD 3 clause

import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import patches

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileTransformer


y_full = target.values

# Take only 2 features to make visualization easier
# AMT_CREDIT has a long tail distribution.
# AMT_INCOME_TOTAL has a few but very large outliers.

X = apps[['AMT_CREDIT', 'AMT_INCOME_TOTAL']].values

distributions = [
    ('Unscaled data', X),
    ('Data after standard scaling',
        StandardScaler().fit_transform(X)),
    ('Data after min-max scaling',
        MinMaxScaler().fit_transform(X)),
    ('Data after max-abs scaling',
        MaxAbsScaler().fit_transform(X)),
    ('Data after robust scaling',
        RobustScaler(quantile_range=(25, 75)).fit_transform(X)),
    ('Data after quantile transformation (uniform pdf)',
        QuantileTransformer(output_distribution='uniform')
        .fit_transform(X)),
    ('Data after quantile transformation (gaussian pdf)',
        QuantileTransformer(output_distribution='normal')
        .fit_transform(X)),
    ('Data after sample-wise L2 normalizing',
        Normalizer().fit_transform(X))
]

y = y_full


def create_axes(title, figsize=(16, 6)):
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)

    # define the axis for the first plot
    left, width = 0.1, 0.22
    bottom, height = 0.1, 0.7
    bottom_h = height + 0.15
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter = plt.axes(rect_scatter)
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)

    # define the axis for the zoomed-in plot
    left = width + left + 0.2
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter_zoom = plt.axes(rect_scatter)
    ax_histx_zoom = plt.axes(rect_histx)
    ax_histy_zoom = plt.axes(rect_histy)

    return ((ax_scatter, ax_histy, ax_histx),
            (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom))


def plot_distribution(axes, X, y, hist_nbins=50, title="",
                      x0_label="", x1_label=""):
    ax, hist_X1, hist_X0 = axes

    ax.set_title(title)
    ax.set_xlabel(x0_label)
    ax.set_ylabel(x1_label)

    # The scatter plot    
    ax.scatter(X[y < 0.5, 0], X[y < 0.5, 1], alpha=0.2, marker='o', s=2, lw=0, c='yellow', label='Target = 0')
    ax.scatter(X[y > 0.5, 0], X[y > 0.5, 1], alpha=0.2, marker='o', s=2, lw=0, c='black', label='Target = 1')
    ax.legend()

    # Removing the top and the right spine for aesthetics
    # make nice axis layout
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    # Histogram for axis X1
    hist_X1.set_ylim(ax.get_ylim())
    hist_X1.hist(X[:, 1], bins=hist_nbins, orientation='horizontal',
                 color='grey', ec='grey')
    hist_X1.axis('off')
    

    # Histogram for axis X0
    hist_X0.set_xlim(ax.get_xlim())
    hist_X0.hist(X[:, 0], bins=hist_nbins, orientation='vertical',
                 color='grey', ec='grey')
    hist_X0.axis('off')


def make_plot(item_idx):
    title, X = distributions[item_idx]
    ax_zoom_out, ax_zoom_in = create_axes(title)
    axarr = (ax_zoom_out, ax_zoom_in)
    plot_distribution(axarr[0], X, y, hist_nbins=200,
                      x0_label="Credit amount",
                      x1_label="Total income",
                      title="Full data")

    # zoom-in
    zoom_in_percentile_range = (0, 99)
    cutoffs_X0 = np.percentile(X[:, 0], zoom_in_percentile_range)
    cutoffs_X1 = np.percentile(X[:, 1], zoom_in_percentile_range)

    non_outliers_mask = (
        np.all(X > [cutoffs_X0[0], cutoffs_X1[0]], axis=1) &
        np.all(X < [cutoffs_X0[1], cutoffs_X1[1]], axis=1))
    plot_distribution(axarr[1], X[non_outliers_mask], y[non_outliers_mask],
                      hist_nbins=50,
                      x0_label="Credit amount",
                      x1_label="Total income",
                      title="Zoom-in")


# ## Standardization: scale *vs* transform
# To metigate the effect of the heavily skewed distribution, the present of outliers, and having different range values, we need to standardize the data.  There are two main ways of doing this.  We can **scale** the data by applying linear tranformations, or we can non-linearly **transform** it.
# 
# In the figures below, we see the joint distribution of the total income and credit amount values colored by default (black dot) and non-default (yellow dot).  The charts are adapted from [this sklearn's tutorial][1].  On the left plot we have a full view of the data where the outlier points compress the rest of the data in a tiny sliver.  The right plot zooms in on that tiny sliver for better illustration.
# 
# [1]: http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py

# In[ ]:


make_plot(0)


# ## Scale to unit variance
# By scaling we remove the mean value and then divide by the standard deviation.  We can achieve this using [sklearn's `StandardScaler`][1].  The data ends up having unit variance, but the effect of the outliers can be seen in the left figure below where most of the data are compressed in s small range.  Furthermore, because the features have different outlier values, they end up having different range values.
# 
# ### Do not use if
# Removing the mean value from all data points destroys the sparseness structure of sparse data.  This type of scaling is also susceptible to the presense of marginal outliers.  There are other types of feature standardization you may consider using if your data is sparse or has a few large marginal outilers.
# 
# [1]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler

# In[ ]:


make_plot(1)


# ## Scale to range
# To metigate the effect of ending up with transformed features having different ranges, we can dictate the range to the [sklearn's `MinMaxScaler`][1].  Our features will be scalled to the given range, but **will not be centered around zero**.
# 
# [`MaxAbsScaler`][2] is another alternative that scales the max absolute value to 1.  However, the lower range of the scaled feature is not guaranteed to be zero.
# 
# ### Do not use if
# As we can see from the plots below, this scaler is affected by a few marginal outliers.  Notice how in the plot on the right below the range for most of the points for the credit amount is [0, 0.4], while it's [-0.002, 0.006] for the total income.
# 
# [1]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
# [2]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html#sklearn.preprocessing.MaxAbsScaler

# In[ ]:


make_plot(2)


# In[ ]:


make_plot(3)


# ## Scale with outliers
# To partially metigate the problem of marginal outliers in the data, we can use the [sklearn's `RobustScaler`][1].  It scales the data to its middle 50% ([IQR][2]) and is thus robust to a few outliers.  We can see in the right chart below how both features have most of their values lie in similar ranges [-1, 3.5] and [-1, 2.5].
# 
# Note that we still have outlier points (left plot below) in both features.  Their negative effect may be felt depending on the machine learning algorithm we later use.
# 
# ### Do not use if
# This scaler removes the mean from the data destroying its sparseness structure.  Avoid using it if your data is sparse.
# 
# [1]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler
# [2]: https://en.wikipedia.org/wiki/Interquartile_range

# In[ ]:


make_plot(4)


# ## Non-linear transform
# Linear transormations preserve the big gap between the marginal outliers and the rest of the data.  However, we can apply a non-linear transformation to change the distribution of the data and bring the outliers to order.  [Sklearn's `QuantileTransformer`][1] molds the values into uniform distribution in the [0, 1] range by default.  It can also be parametrized to transform the values into a Gaussian distribution.
# 
# ### Do not use if
# This non-linear transformation may destroy the feature's linear correlation with other features.  It also desroys the sparseness structure of the data. Consider using other feature scaling or transformers if feature linear correlation or sparseness structure needs to be preserved.
# 
# [1]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer

# In[ ]:


make_plot(5)


# In[ ]:


make_plot(6)


# ## Normalization of rows
# All previously discussed scalers and transformers standardize the features independently.  The total income feature was scaled and sometimes centered independent of the credit amount feature.  If the features are related or are used later on to derive a variable then we may want to normalize them row-wise to unit norm.  This transformation resuls in having our data points with unit distance from the origin.
# 
# ### Do not use if
# If the features are not related or have different types, then you should probably use a different standardization measure.

# In[ ]:


make_plot(7)

