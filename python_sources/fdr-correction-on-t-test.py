#!/usr/bin/env python
# coding: utf-8

# Learned from: https://mne.tools/stable/auto_examples/stats/plot_fdr_stats_evoked.html
# 
# One tests if the evoked response significantly deviates from 0. Multiple comparison problem is addressed with False Discovery Rate (FDR) correction.
# 
# Reject here means rejecting the null hypothesis => True => if it not right, then it is a false positive (accepting the impact though the two are just simply coming from the same distribution)

# In[ ]:


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# In[ ]:


import mne
from mne import io
from mne.datasets import sample
from mne.stats import bonferroni_correction, fdr_correction

print(__doc__)


# In[ ]:


data_path = sample.data_path()


# In[ ]:


raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
event_id, tmin, tmax = 1, -0.2, 0.5


# In[ ]:


# Setup for reading the raw data
raw = io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)[:30]

channel = 'MEG 1332' # include only this channel in analysis
include = [channel]


# Read epochs for the channel of interest

# In[ ]:


picks = mne.pick_types(raw.info, meg=False, eog = True, include=include, exclude='bads')


# In[ ]:


event_id = 1
reject = dict(grad=4000e-13, eog=150e-6)


# In[ ]:


epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks, baseline=(None, 0), reject=reject)


# In[ ]:


X = epochs.get_data() # as 3D matrix
X = X[:, 0, :] # take only one channel to get a 2D array


# Compute statistics

# In[ ]:


get_ipython().run_line_magic('pinfo', 'stats.ttest_1samp')


# In[ ]:


T, pval = stats.ttest_1samp(X, 0)


# In[ ]:


X.shape


# In[ ]:


len(T)


# In[ ]:


len(pval)


# In[ ]:


len(X)


# In[ ]:


#Check to understand the calculation from here: https://www.socscistatistics.com/tests/studentttest/default2.aspx
print(stats.ttest_ind(a=[[20, 25], [0.3, 0.4]], b=[[27, 28], [0.1, 0.2]]))


# In[ ]:


print(stats.ttest_ind(a=[[20, 25], [0.3, 0.4]], b=[[27, 28], [0.1, 0.2]], axis=1))


# In[ ]:


alpha = 0.05


# In[ ]:


n_samples, n_tests = X.shape


# In[ ]:


threshold_uncorrected = stats.t.ppf(1.0 - alpha, n_samples - 1)


# In[ ]:


threshold_uncorrected


# In[ ]:


reject_bonferroni, pval_bonferroni = bonferroni_correction(pval, alpha=alpha)
threshold_bonferroni  = stats.t.ppf(1.0-alpha/n_tests, n_samples - 1)


# In[ ]:


threshold_bonferroni


# In[ ]:


reject_fdr, pval_fdr = fdr_correction(pval, alpha=alpha, method='indep')
threshold_fdr = np.min(np.abs(T)[reject_fdr])


# In[ ]:


threshold_fdr


# In[ ]:


reject_fdr


# # Plot

# In[ ]:


times = 1e3*epochs.times
plt.close('all')
plt.plot(times, T, 'k', label='T-stat')
xmin, xmax = plt.xlim()
plt.hlines(threshold_uncorrected, xmin, xmax, linestyle='--', colors='k', label='p=0.05 (uncorrected)', linewidth = 2)
plt.hlines(threshold_bonferroni, xmin, xmax, linestyle='--', colors='r', label='p=0.05 (Bonferroni)', linewidth = 2)
plt.hlines(threshold_fdr, xmin, xmax, linestyle='--', colors='b', label='p=0.05 (FDR)', linewidth = 2)
plt.plot(times, pval, 'o', label='Pval')
plt.plot(times, pval_fdr, 'v', label='pval_fdr')
plt.legend()
plt.xlabel("Time (ms)")
plt.ylabel('T-stat')
plt.show()


# In[ ]:


pval_fdr < 0.05


# In[ ]:


reject_fdr


# In[ ]:


pval < 0.05


# In[ ]:




