#!/usr/bin/env python
# coding: utf-8

# # Kernel density estimation with TED talks
# 
# The central orginazatory definition of probability theory is the idea of density: that, for a given (continuous) distribution, the probability that a value will fall in between so-and-so and that-and-that is such-and-such number. By considering as many dimensions in this density as we have dimensions in the data, we arrive at a probabilistic model for what our data is doing.
# 
# **Kernel density estimates** are a family of methods for estimating these underlying densities. In this notebook I'll take a quick look at their properties, and discuss why they are useful, using as a gineau pig TED Talk lengths.
# 
# ## Munging the data

# In[ ]:


import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import pandas as pd
df = pd.read_csv("../input/ted-talks/ted_main.csv")
df.head()


# ## Estimating density
# 
# Kernel density estimation is useful in particular because it is *non-parametric*. What that means is that KDEs do not rely on any assumptions about the shape of the underlying data; they are based instead solely on observed properties of the data. This is very different from, say, a linear regression model, which makes *lots* of (so-called "parametric") assumptions about the data: that residuals are heteroskedaic, for example; or better yet, that the relationship in the data is linear at all.
# 
# To demonstrate, let's look at the simplest non-parametric density estimation method: a histogram.

# In[ ]:


df['duration'].plot.hist(20)


# In this plot we see that the vast majority of TED Talks are under 1200 seconds (20 minutes), peaking at around the perhaps-18-minutes mark. It's a bit difficult to make this out exactly, however. In general histograms are a very useful exploratory technique, at least from the perspective of "first approximation". However, they suffer from two major shortcomings. The first is that the "look" of a histogram depends a *lot* on where you place the bin edges (they tend to work well for normal distributions, but to fall apart for more complicated things). The second is that it requires a lot of samples, and furthermore requires roughly doubly the number of samples to get roughly twice the accuracy.
# 
# These limitations aren't ideal because they make the data harder to interpret. A KDE smooths over them by *interpolating*: by coming up with logical values to "fill in" gaps between points in the data. For example, here is the KDE plot for this data (using `seaborn`):

# In[ ]:


import seaborn as sns
sns.kdeplot(df['duration'])


# We can see how much clearer this view of the data is. It shows that TED talks actually peak in duration at around 1000 seconds (that's ~16.5 minutes); we can even see what might be an interesting substructure at around the five-minute mark.
# 
# Kernel density estimates do this by (effectively) computing a density for every space of points in the distribution. That density is a function of the data points that are near that point in space. Roughly speaking, data points near the theoretical point under consideration are given a lot of weight towards determining the value of that point, while data points further away are given less value. After a certain "cutoff", points no longer have any influence on the density estimate.
# 
# The points that end up being used for the estimate are blended together using a function known as a **kernel**. The cutoff of points used to build the estimate is controlled by a hyperparameter known as the **bandwidth**. By using a kernel with a given hyperparameter to make an estimate of density at every point of interest on the curve, we build a *kernel density estimate* for the entire distribution.
# 
# So for example, if we took a point (say, 1300 seconds), took every point within 100 seconds of that point (so a bandwidth of 50? or 100), and took the average of those points, that would be an example of a simple kernel density estimate.
# 
# In practice, the simplest kernel that tends to get used is the Gaussian kernel. An example of that is above, but here's another one that's more demonstrative of the effect that bandwidth has on the output.

# In[ ]:


import numpy as np
X = df['duration'].values[:, np.newaxis]
from sklearn.neighbors.kde import KernelDensity
kde10 = KernelDensity(kernel='gaussian', bandwidth=10)
kde10.fit(X)
kde40 = KernelDensity(kernel='gaussian', bandwidth=40)
kde40.fit(X)
kde100 = KernelDensity(kernel='gaussian', bandwidth=100)
kde100.fit(X)


# In[ ]:


n2000 = np.array(list(range(0, 2001)))[:, np.newaxis]

fig, ax = plt.subplots(1, figsize=(12, 6))
ax.plot(n2000, np.exp(kde10.score_samples(n2000)), label='10-band')
ax.plot(n2000, np.exp(kde40.score_samples(n2000)), label='40-band')
ax.plot(n2000, np.exp(kde100.score_samples(n2000)), label='100-band')
ax.legend()
ax.set_title("TED Talk Speech Lengths, KDE Estimates by Bandwidth (Gaussian)")


# The blue KDE is a Gaussian estimate with a bandwidth of 10. The orange KDE is a Gaussian estimate with a bandwidth of 40. The decision on what bandwidth is best is a perhaps somewhat arbitrary user choice: the 10-bandwidth KDE shows us more intereszting features of the data (like the fact that speeches are significantly more likely to be 10 minutes long than 9 minutes long), but also demonstrates more variance than the far smoother 40-bandwidth KDE.
# 
# In fact, this is what choosing a bandwidth is all about: making a trade-off between [bias and variance](https://www.kaggle.com/residentmario/bias-variance-tradeoff). The 100-bandwidth curve, for example, is starting too look *too* biased: it misses, for example, the height of the "peak" of the data around the 1000-second mark.
# 
# Note that KDE produces probability distributions, not functions. A way of recovering curves that look a lot like these that are nevertheless functional is [polynomial regression](https://www.kaggle.com/residentmario/pumpkin-price-polynomial-regression/) (or, more flexibly but less functionally, a [piecewise spline](https://datascience.stackexchange.com/questions/8457/python-library-for-segmented-regression-a-k-a-piecewise-regression)).
# 
# Although Gaussian kernels are the most common one used by far (and is generally the most appropriate kernel for normal-ish data), others exist. For example, the `scikit-learn` documentation posits the tophat KDE as an alternative to the histogram:

# In[ ]:


kde10tophat = KernelDensity(kernel='tophat', bandwidth=10).fit(X)
kde40tophat = KernelDensity(kernel='tophat', bandwidth=40).fit(X)

fig, ax = plt.subplots(1, figsize=(12, 6))
ax.plot(n2000, np.exp(kde10tophat.score_samples(n2000)), label='10-band')
ax.plot(n2000, np.exp(kde40tophat.score_samples(n2000)), label='40-band')
ax.legend()
ax.set_title("TED Talk Speech Lengths, KDE Estimates by Bandwidth (Tophat)")


# In practice, the Gaussian KDE plot gets used as a replacement for the histogram.
# 
# [This page](http://scikit-learn.org/stable/modules/density.html) and especially [this plot](http://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html) for the `sklearn` documentation has more details on kernel choices.
# 
# ## Reconstructing data
# 
# The other interesting fold to using KDE is that since it generates a probability distribution, it can be used to create new sample data based on existing samples. This can be done directly using the `sample` method. The `sklearn` documentation has the following fascinating code sample demonstrating this in action:

# In[ ]:


from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

# load the data
digits = load_digits()
data = digits.data

# project the 64-dimensional data to a lower dimension
pca = PCA(n_components=15, whiten=False)
data = pca.fit_transform(digits.data)

# use grid search cross-validation to optimize the bandwidth
params = {'bandwidth': np.logspace(-1, 1, 20)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit(data)

print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

# use the best estimator to compute the kernel density estimate
kde = grid.best_estimator_

# sample 44 new points from the data
new_data = kde.sample(44, random_state=0)
new_data = pca.inverse_transform(new_data)

# turn data into a 4x11 grid
new_data = new_data.reshape((4, 11, -1))
real_data = digits.data[:44].reshape((4, 11, -1))

# plot real digits and resampled digits
fig, ax = plt.subplots(9, 11, subplot_kw=dict(xticks=[], yticks=[]))
for j in range(11):
    ax[4, j].set_visible(False)
    for i in range(4):
        im = ax[i, j].imshow(real_data[i, j].reshape((8, 8)),
                             cmap=plt.cm.binary, interpolation='nearest')
        im.set_clim(0, 16)
        im = ax[i + 5, j].imshow(new_data[i, j].reshape((8, 8)),
                                 cmap=plt.cm.binary, interpolation='nearest')
        im.set_clim(0, 16)

ax[0, 5].set_title('Selection from the input data')
ax[5, 5].set_title('"New" digits drawn from the kernel density model')

plt.show()


# It's remarkable how well this works. Most of the newly generated digits are clearly identifiable.
# 
# This is an unsupervised technique, e.g. one that's agnostic to class labels. It works best when done with preprocessing with PCA, which "cleans up" the data vectors and helps defend against the effect of variance on the samples. It requires relatively clean data with reasonably distinct classes. It works well on MNIST data because MNIST data is so simple; I'm not sure how well it will generalize to other kinds of data. However, it is nevertheless a very powerful technical idea!
