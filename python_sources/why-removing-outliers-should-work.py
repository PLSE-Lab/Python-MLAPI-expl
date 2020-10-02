#!/usr/bin/env python
# coding: utf-8

# I am going to describe picturially why removing outliers should work on this dataset. 
# 
# First I generate a 2-dimensional classification dataset and plot 1, 2, 3 standard deviation of their gaussian distributions based on means and covariance of each class (positive and negative). 

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics, preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import NuSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn import svm, neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.covariance import EllipticEnvelope


# In[ ]:


np.random.seed(2019)
N_points = 512
flip_y = 0.025


def plot_ellipses(inpX):
    cov = np.cov(inpX[:,0],inpX[:, 1])

    for nstd in [1, 2, 3]:
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        w, h = 2 * nstd * np.sqrt(vals)
        ell = Ellipse(xy=(np.mean(inpX[:, 0]), np.mean(inpX[:, 1])),
                      width=w, height=h,
                      angle=theta, color='black', alpha=1/nstd)
        ell.set_facecolor('none')
        ax.add_artist(ell)

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

X = np.random.randn(N_points, 2)
X_pos = X[:N_points//2].dot(np.array([[.6, -.6],
                                        [-.15, 1.5]]))  + np.array([1.5, 1.5])

X_neg = X[N_points//2:] #- np.array([1, 1])
y_pos = np.ones(len(X_pos))
y_neg = np.zeros(len(X_neg))

plt.figure(figsize=(10,10))
ax = plt.subplot(111)

ax.scatter(X_pos[:, 0], X_pos[:, 1], c='red', label='positive')
ax.scatter(X_neg[:, 0], X_neg[:, 1], c='blue', label='negative')


plot_ellipses(X_pos)
plot_ellipses(X_neg)
plt.xlim([-4, 4])
plt.ylim([-3, 6])
plt.legend()


# These ellipses are more or less the true distributions we are challenged to find in this competition. Note that each chunk of our dataset have 33 to 47 dimensions so we have hyper ellipses.
# 
# Now, I generate the same thing but this time I flip some of the targets (for the sake of illustration I only flipped some of the negatives to positive and not vice versa). This is similar to what `sklearn.make_classification` does by assigning random classes to some points.

# In[ ]:


np.random.seed(2019)

X = np.random.randn(N_points, 2)
X_pos = X[:N_points//2].dot(np.array([[.6, -.6],
                                        [-.15, 1.5]]))  + np.array([1.5, 1.5])

X_neg = X[N_points//2:] #- np.array([1, 1])
y_pos = np.ones(len(X_pos))
y_neg = np.zeros(len(X_neg))

mask_id = np.random.randint(0,N_points//2+1,int(flip_y*N_points))
X_pos_holdout = X_pos[mask_id, :].copy()
X_pos[mask_id, :] = X_neg[mask_id, :]
y_pos[mask_id] = 0
X_neg[mask_id, :] = X_pos_holdout
X_neg[mask_id] = 1

X = np.vstack((X_pos, X_neg))
y = np.concatenate((y_pos, y_neg))


plt.figure(figsize=(10,10))
ax = plt.subplot(111)

ax.scatter(X_pos[:, 0], X_pos[:, 1], c='red', label='positive')
ax.scatter(X_neg[:, 0], X_neg[:, 1], c='blue', label='negative')


plot_ellipses(X_pos)
plot_ellipses(X_neg)
plt.xlim([-4, 4])
plt.ylim([-3, 6])
plt.legend()


# Notice how our best fit gaussian distribution is different than the true distribution shown above. 
# 
# Looking at these plots, I believe finding outliers and removing them should help us find parameters of gaussian distributions better and improve our score.
# 
# How can we detect outliers? Watch https://www.coursera.org/lecture/machine-learning/anomaly-detection-using-the-multivariate-gaussian-distribution-DnNr9
# 
# Considering that we know our classes are from multivariate gaussian distributions, and by looking at our plots above one way would be to remove anything that falls outside our 3 standard deviation ellipse. 
# 

# In[ ]:



np.random.seed(2019)
N_points = 512
flip_y = 0.025

X = np.random.randn(N_points, 2)
X_pos = X[:N_points//2].dot(np.array([[.6, -.6],
                                        [-.15, 1.5]]))  + np.array([1.5, 1.5])

X_neg = X[N_points//2:] #- np.array([1, 1])
y_pos = np.ones(len(X_pos))
y_neg = np.zeros(len(X_neg))

mask_id = np.random.randint(0,N_points//2+1,int(flip_y*N_points))
X_pos_holdout = X_pos[mask_id, :].copy()
X_pos[mask_id, :] = X_neg[mask_id, :]
y_pos[mask_id] = 0
X_neg[mask_id, :] = X_pos_holdout
X_neg[mask_id] = 1

X = np.vstack((X_pos, X_neg))
y = np.concatenate((y_pos, y_neg))


plt.figure(figsize=(10,10))
ax = plt.subplot(111)

ax.scatter(X_pos[:, 0], X_pos[:, 1], c='red', label='positive')
ax.scatter(X_neg[:, 0], X_neg[:, 1], c='blue', label='negative')

def plot_ellipses(inpX):
    cov = np.cov(inpX[:,0],inpX[:, 1])

    for nstd in [1, 2, 3]:
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        w, h = 2 * nstd * np.sqrt(vals)
        ell = Ellipse(xy=(np.mean(inpX[:, 0]), np.mean(inpX[:, 1])),
                      width=w, height=h,
                      angle=theta, color='black')
        ell.set_facecolor('none')
        ax.add_artist(ell)
        
EE = EllipticEnvelope(contamination=0.016) # notice that I did it manually to find only those that fall outside 3 std. I have a lot of kernels and couldn't find the code I was using for this method :D so please accept this for now. 
EE.fit(X_pos)
pos_ids = (EE.predict(X_pos) == -1)

plt.scatter(X_pos[pos_ids, 0], X_pos[pos_ids, 1], color='yellow', label='outliers')

plot_ellipses(X_pos)
plot_ellipses(X_neg)
plt.xlim([-4, 4])
plt.ylim([-3, 6])
plt.legend()


# Another way is to use `sklearn.covariance.EllipticEnvelope()` and use `contamination==0.025` argument which is "The amount of contamination of the data set, i.e. the proportion of outliers in the data set."

# In[ ]:



np.random.seed(2019)
N_points = 512
flip_y = 0.025

X = np.random.randn(N_points, 2)
X_pos = X[:N_points//2].dot(np.array([[.6, -.6],
                                        [-.15, 1.5]]))  + np.array([1.5, 1.5])

X_neg = X[N_points//2:] #- np.array([1, 1])
y_pos = np.ones(len(X_pos))
y_neg = np.zeros(len(X_neg))

mask_id = np.random.randint(0,N_points//2+1,int(flip_y*N_points))
X_pos_holdout = X_pos[mask_id, :].copy()
X_pos[mask_id, :] = X_neg[mask_id, :]
y_pos[mask_id] = 0
X_neg[mask_id, :] = X_pos_holdout
X_neg[mask_id] = 1

X = np.vstack((X_pos, X_neg))
y = np.concatenate((y_pos, y_neg))


plt.figure(figsize=(10,10))
ax = plt.subplot(111)

ax.scatter(X_pos[:, 0], X_pos[:, 1], c='red', label='positive')
ax.scatter(X_neg[:, 0], X_neg[:, 1], c='blue', label='negative')

def plot_ellipses(inpX):
    cov = np.cov(inpX[:,0],inpX[:, 1])

    for nstd in [1, 2, 3]:
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        w, h = 2 * nstd * np.sqrt(vals)
        ell = Ellipse(xy=(np.mean(inpX[:, 0]), np.mean(inpX[:, 1])),
                      width=w, height=h,
                      angle=theta, color='black')
        ell.set_facecolor('none')
        ax.add_artist(ell)
        
EE = EllipticEnvelope(contamination=0.025)
EE.fit(X_pos)
pos_ids = (EE.predict(X_pos) == -1)

plt.scatter(X_pos[pos_ids, 0], X_pos[pos_ids, 1], color='yellow', label='outliers')

plot_ellipses(X_pos)
plot_ellipses(X_neg)
plt.xlim([-4, 4])
plt.ylim([-3, 6])
plt.legend()


# ### At the end of the day, removing outliers didn't work for me. I am still wondering why?
# 
# ### I appreciate your ideas. Please share.
