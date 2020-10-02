#!/usr/bin/env python
# coding: utf-8

# # ML visualization with yellowbrick (1)
# 
# This is the first notebook in short series in which I explore the `yellowbrick` library.
# 
# The cute module name comes from the famous "follow the yellow brick road" line in The Wizard of Oz. Data visualization is an integral part of the machine learning process during all of the preprocessing, exploration, and model-building stages. `yellowbrick` is the only machine learning -focused visualization library that I'm aware of, which probably means it's the only one with any real traction. This is surprising of course; it turns out that a lot of ML engineers are happy to rely on hacky `matplotlib` recipes they once spent a lot of time working out! An ML process -focused data viz library sounds like a honking good idea.
# 
# Let's see how well this library executes on the promise. Note that this is all based on the official documentation [here](http://www.scikit-yb.org/en/latest/api/index.html).

# ## Data
# 
# For this notebook I'll use the Kepler space observations dataset.

# In[ ]:


import yellowbrick as yb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pd.set_option('max_columns', None)

df = pd.read_csv("../input/cumulative.csv")

X = (df.iloc[:, 5:].assign(koi_pdisposition=(df.koi_pdisposition == "CANDIDATE").astype(int)).drop(columns=['koi_tce_delivname']))
y = df.koi_disposition == "CONFIRMED"

df.head()


# ## Visualizations
# 
# ### Rank1D
# 
# This plots a simple bar chart in one dimension ranking the data according to some statistic, hence the name. The name is a misnomer for now, however, as though the method provides an `algorithm` field the only algorithm that works at the moment is the Shapiro-Wilk test, a statistical test (with a confidence score on the $[0, 1]$ scale) of whether or not the given feature is normally distributed. The Shapiro-Wilk is a useful tool for visualizations because it lets you test normality of variables using a single test statistic, e.g. without relying on a huge list of plots. Much less cumbersome. For more on the test, [the Wikipedia article](https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test) is informative. If you're not familiar with hypothesis testing, I wrote [a notebook on this subject](https://www.kaggle.com/residentmario/hypothesis-testing-with-firearm-licensees/) which might be worth reading.
# 
# Here's what you get:

# In[ ]:


from yellowbrick.features import Rank1D

fig, ax = plt.subplots(1, figsize=(8, 12))
vzr = Rank1D(ax=ax)
vzr.fit(X, y)
vzr.transform(X)
sns.despine(left=True, bottom=True)
vzr.poof()


# This is the output of a statistical test, but since we're not making any decisions based on it there's no need to choose a specific level of significance ($p$ value). You can see that `ra` and `dec` are very normally distributed ($p = 0.05$). The rest of the values are not normally distributed. Note that those flag values, which may only be 0 or 1, are scoring between 0.5 and 0.6 according to this metric! Semi-interesting.
# 
# It's dissappointing that the Shapiro is the only algorithm you can currently output to this plot. We need more testing options, and also the option to provide the test statistic yourself!

# ## Rank2D
# 
# Rank2D implements a `seaborn` heatmap (reference to that [here](https://seaborn.pydata.org/examples/heatmap_annotation.html)) with some good defaults. Where `Rank1D` relies on one-dimensional metrics, `Rank2D` provides a facility for two-dimensional metrics, e.g. things like correlation, covariance, and so on.

# In[ ]:


from yellowbrick.features import Rank2D

fig, ax = plt.subplots(1, figsize=(12, 12))
vzr = Rank2D(ax=ax)
vzr.fit(X, y)
vzr.transform(X)
sns.despine(left=True, bottom=True)
vzr.poof()


# Again the API is super-limiting, with only `covariance` and `pearson` options (see [this notebook](https://www.kaggle.com/residentmario/pearson-s-r-with-health-searches) for a background on these metrics). There are lots more useful metrics that could be implemented; Spearmen's correlation, for example (I discuss that [here](https://www.kaggle.com/residentmario/spearman-correlation-with-montreal-bikes/)). This API needs expanding!
# 
# Nevertheless, for that particular use case this is a lot more convenient than specifying a heatmap manually.

# ## Parallel Coordinates
# 
# Parallel coordinates are a lovely plot type. Given an input of normalized data, parallel coordinate plotting will lay out the feature space and plot out where each point falls on it.

# In[ ]:


from yellowbrick.features import ParallelCoordinates
from sklearn.preprocessing import StandardScaler

# Select a handful of relevant columns and drop nulls.
np.random.seed()
cols = ['koi_pdisposition', 'koi_score', 'koi_duration', 'ra', 'dec']
X_sample = X.sample(500).loc[:, cols].dropna()
y_sample = y.iloc[X_sample.index.values].reset_index(drop=True)

# Normalize all of the fields.
trans = StandardScaler()
trans.fit(X_sample)
X_sample = pd.DataFrame(trans.transform(X_sample), columns=cols)

# Fit the chart.
# fig, ax = plt.subplots(1, figsize=(12, 6))
kwargs = {'vlines_kwds': {'color': 'lightgray'}}
vzr = ParallelCoordinates(classes=['NOT CONFIRMED', 'CONFIRMED'], **kwargs)  # ax=ax
vzr.fit(X_sample, y_sample)
vzr.transform(X_sample)
sns.despine(left=True, bottom=True)

# Display.
vzr.poof()


# We see a few different interesting effects here. For one there's a long push of outliers of observation durations, all of which correspond with observations that were `NOT CONFIRMED`. That seems significant. Also, we can see that the probability that Kepler assigns to planets that get `CONFIRMED` (`koi_score`) is quite high, but not always 100 percent (in interpreting this fact, recall that this feature is scaled to fall between $[-1, 1]$).
# 
# Parallel coordinates plots are overall a very useful chart type, though not one without weaknesses. It's great to have a neat interface to it like this. But it's worth pointing out that this plot type is just reimplementing a `pandas.plotting` built-in (see [here](https://pandas.pydata.org/pandas-docs/stable/visualization.html)). Meanwhile another related albeit less interpretable `pandas.plotting` built-in, the Andrews curve, is missing.
# 
# For a bit more on parallel coordinates I have a brief section on it in the Python Data Visualization Learn Track ([here](https://www.kaggle.com/residentmario/multivariate-plotting)).

# ### RadViz
# 
# `RadViz` is another re-packaged `pandas.plotting` built-in. This visualization type lays out the features of the dataset in a circle, then plots the position of each point under consideration in that circle by pushing it towards the variables it loads heavily in. This is also sometimes called the "spring layout". It can be quite useful for visualizing distinguishabile attributes between class clusters; I find it to be a very understandable way of explaining which $n$ variables a particular class loads heavily on, when $n$ is greater than 2.

# In[ ]:


from yellowbrick.features import RadViz

# fig, ax = plt.subplots(1, figsize=(12, 6))
# cmap = y_sample.map(lambda v: "steelblue" if v else "lightgray")
vzr = RadViz(classes=['NOT CONFIRMED', 'CONFIRMED'])
vzr.fit_transform(X_sample, y_sample)
vzr.poof()


# This chart only tells us that planets that got confirmed loaded heavily in the pre-disposition, which is something we probably already figured.

# ### PCADecomposition
# 
# PCA, or Principal Components Analysis, is a dimensionality reduction technique which lets us drop the number of variables under consideration to some user-specified subset thereof. PCA works by finding the "natural directions" in a dataset, e.g. calculating new synthetic feature observations along the dataset axes that cover the most variance in the dataset, and hence, are the most "interesting". Note that PCA is an unsupervised algorithm that does not consider labels we assign to the data (e.g. classes). I wrote a lengthy primer on PCA that you can read [here](https://www.kaggle.com/residentmario/dimensionality-reduction-and-pca-for-fashion-mnist/) if you're not already familiar with it.
# 
# PCA is a great technique for EDA because by examing the axes that get chosen by the algorithm we're better able to understand what combinations of variables have the highest variance and "coverage" in the dataset. What does `yellowbrick` do with this?

# In[ ]:


from yellowbrick.features import PCADecomposition

fig, ax = plt.subplots(1, figsize=(12, 6))
cmap = y_sample.map(lambda v: "steelblue" if v else "lightgray")
vzr = PCADecomposition(color=cmap)
vzr.fit_transform(X_sample, y_sample)
vzr.poof()


# In[ ]:


from yellowbrick.features import PCADecomposition
vzr = PCADecomposition(proj_dim=3)
vzr.fit_transform(X_sample, y_sample)
vzr.poof()


# These 2-d and 3-d scatter plots are good for probing how difficult a classification or regression problem will be. Here we see that the observations naturally cluster into two groups, but that the relative distribution of the class of interest *within* that group doesn't differ between them. If we move on to examining what it is about the data that is creating these quite separable structures, we will make great gains in understanding what the underlying data describes.
# 
# There is more that can be done with PCA than this, though. This library needs more options!

# ### FeatureImportances
# 
# Feature importance is the relative usefulness of a given feature for performing a classification or regression task. The `FeatureImportances` chart type takes advantage of the exploratory power of decision tree algorithms, which provide a `feature_importance_` result once fitted, to plot this information directly. If you're not familiar with decision trees you can read up on them [here](https://www.kaggle.com/residentmario/decision-trees-with-animal-shelter-outcomes).

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from yellowbrick.features import FeatureImportances

clf = DecisionTreeClassifier()
viz = FeatureImportances(clf)
viz.fit(X_sample, y_sample)
viz.poof()


# This plot type is a great use of decision trees for EDA. Again though, you can go a bit further by training and fiddling with the decision tree yourself.
# 
# # Conclusion
# 
# These are all of the feature exploration features included in `yellowbrick`. That concludes this notebook! For the next section click here: https://www.kaggle.com/residentmario/ml-visualization-with-yellowbrick-2/
