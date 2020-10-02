#!/usr/bin/env python
# coding: utf-8

# # Decoding Visual Imagination from MEG data: Time-resolved vs. Covariance
# 
# A person imagines a familiar face or location (`y`) for 6 seconds while the brain activity is recorded from 306 sensors (`X`) with an MEG (magnetoencephalography) device. This is repeated 100-200 times (`n_trials`). The (concatenated) timeseries of one trial can be fed to a classifier to predict what was imagined - face or location - from just the MEG signal. Usually this approach, called time-resolved decoding, does not work. An alternative approach is to compute the covariance matrix between the sensors for each trial and then to predict what was imagined in that trial from the covariance matrix. Notice that covariance matrices live on a Riemannian manifold, so the distance between two covariance matrices is the Riemannian distance. We use [pyRiemann](https://github.com/alexandrebarachant/pyRiemann) to compute such quantities and do classification.
# 

# ## Loading and preparing data

# In[ ]:


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

print("Loading and extracting data")
filename = 'subject_01.npz'
print(f"Loading {filename}")
data = np.load('/kaggle/input/visual-imagination-with-meg/' + filename)
X = data['X']  # timeseries: n_trials x n_sensors x n_timesteps
y = data['y']  # class labels: n_trials
print(f"X: {X.shape}")
print(f"y: {y.shape}")
print(f"n_trials: {X.shape[0]}")

t_min = 2.0  # sec.
t_max = 4.0  # sec.
freq = 250.0  # Hz

print(f"Extracting timeseries from {t_min}sec to {t_max}sec at {freq}Hz")
idx_min = int(t_min * freq)
idx_max = int(t_max * freq)
X_window = X[:, :, idx_min:idx_max]  # we use data from t_min to t_max
print(f"X_window: {X_window.shape}")


# ## Time-resolved decoding

# In[ ]:


print("Concatenating the time-series of each trial into a vector")
X_window_time = X_window.reshape(X_window.shape[0], -1)
print(f"X_window_time: {X_window_time.shape}")

clf = make_pipeline(StandardScaler(), LogisticRegression())
cv = StratifiedKFold(n_splits=4)

print("Predicting what was imagined from the timeseries")
results_time = cross_validate(clf, X=X_window_time, y=y, cv=cv, n_jobs=-1)
print(f"Accuracy: {results_time['test_score'].mean()}")


# ## Covariance-based decoding

# In[ ]:


print("Installing pyRiemann to use Covariances and Riemannian distances on timeseries")
get_ipython().system('pip install pyriemann')


# In[ ]:


from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

print("Creating a pipeline that computes the (regularized) covariance matrix of each trial...")
cov = Covariances(estimator='oas')
print("...and approximates each covariance matrix into a vector through the Tangent Space...")
print("...so that the Euclidean distance between vectors approximates the Riemannian distance between covariance matrices...")
ts = TangentSpace()
print("...and feeds the vectors into a classifier")
clf = make_pipeline(cov, ts, LogisticRegression())

print("Predicting what was imagined from the timeseries")
results_covariance = cross_validate(clf, X=X_window, y=y, cv=cv, n_jobs=-1, verbose=True)
print(f"Accuracy: {results_covariance['test_score'].mean()}")


# In[ ]:




