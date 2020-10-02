#!/usr/bin/env python
# coding: utf-8

# Import libraries we used in this kernel.

# In[ ]:


import numpy as np
from sklearn.svm import SVC
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets.samples_generator import make_circles
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


# Generate training samples. Here we generate two clusters of points and will try to classify them later.

# In[ ]:


samples, labels = make_blobs(n_samples=200, centers=2,
                  random_state=0, cluster_std=0.50)


# Let's see what we have here using the scatter plot.

# In[ ]:


plt.scatter(samples[:, 0], samples[:, 1], c=labels);


# Ok so we have two classes of samples. We will try to classify them using SVM Classifier. Scikit-learn (sklearn for short) library have built-in function **SVC** to help us with the task.
# 
# I will try SVC with a linear kernel because we can see the data classes are linear separatable.
# Call *fit* to fit the model to this dataset.

# In[ ]:


clf = SVC(kernel='linear')
clf.fit(samples, labels)


# Okay so now we have a function to plot the SVC decision boundary and the margins.

# In[ ]:


def plot_svc_decision_function(clf, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 2000)
    y = np.linspace(ylim[0], ylim[1], 2000)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = clf.predict(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contourf(X, Y, P, levels=2, alpha=0.2)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# Let plot the trained SVC model with data points to see if it's doing well.

# In[ ]:


plt.scatter(samples[:, 0], samples[:, 1], c=labels)
plot_svc_decision_function(clf);


# Okay. That's enough for **linear separatable** classes. Now try some data that is **not linear separatable** (can not be separated with a straight line), i.e surrounded classes.

# We can also use the sklearn data generator to generate sample data.

# In[ ]:


samples, labels = make_circles(200, factor=0.1, noise=0.1)


# Now use the scatter plot again to see what we have.

# In[ ]:


plt.scatter(samples[:, 0], samples[:, 1], c=labels);


# For once, try our luck with the linear kernel SVC. Let's try to fit the SVC model with **this new data**.

# In[ ]:


clf.fit(samples, labels)


# Again, scatter plot and decision boundary.

# In[ ]:


plt.scatter(samples[:, 0], samples[:, 1], c=labels)
plot_svc_decision_function(clf);


# Well, bad luck this time. It's when the non-linear kernels come to the rescue. Let's try separating data points in a higher dimension space using the *Radial Basic Function* aka **'rbf'** kernel.

# In[ ]:


rbf_kernel_clf = SVC(kernel='rbf')
rbf_kernel_clf.fit(samples, labels)


# Let's see if this time it can do any better.

# In[ ]:


plt.scatter(samples[:, 0], samples[:, 1], c=labels)
plot_svc_decision_function(rbf_kernel_clf);


# Well, ways better now. But **how**?

# Turns out the 'rbf' kernel is a kind SVC kernel to map 2D data into a 3D space, and the data samples are linear separatable in the 3d space.

# For example, here we can use a kernel function that add a **r** dimension to data and the value of **r** depends on how far **r** is from a point C in space (here C equals (0,0))

# **r** can be calculated as:

# In[ ]:


r = np.exp(-(samples ** 2).sum(1))


# Now we can plot a 3D figure to see if the **r** dimension really helps.

# In[ ]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(samples[:,0], samples[:,1], r, 'o', markersize=8, alpha=0.5)
plt.show()


# Now they are linear separatable ;)

# In[ ]:




