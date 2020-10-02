#!/usr/bin/env python
# coding: utf-8

# 
# 
# * https://www.kaggle.com/tunguz/rapids-ensemble-for-trends-neuroimaging/output [submission_rapids_ensemble.csv] # LB: 0.1595
# * https://www.kaggle.com/moradnejad/trends-eda-fe-mysubmission/output [sub.csv] # LB: 0.1593
# * https://www.kaggle.com/hamditarek/trends-neuroimaging-blend/output [submission.csv] # LB: 0.1595
# * https://www.kaggle.com/aerdem4/rapids-svm-on-trends-neuroimaging/output [submission1.csv] # LB: 0.1598
# * [submission_ridge.csv] # LB: 0.1595

# In[ ]:


import numpy as np
import pandas as pd

# visualize
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from matplotlib import pyplot
from matplotlib.ticker import ScalarFormatter
sns.set_context("talk")
style.use('fivethirtyeight')

from scipy import stats
from tqdm import tqdm_notebook as tqdm
import os


# ## Reading all files

# In[ ]:


# load submission files

BASE_PATH = "../input/trends-assessment-prediction"
SUBMISSION_FILES_PATH = "../input/trends-master-ensemble"

submission = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")

# high scoring public kernels
ENSEMBLES = [
    {"file0": f"{SUBMISSION_FILES_PATH}/submission_rapids_ensemble.csv", "weight": 1.1},
    {"file1": f"{SUBMISSION_FILES_PATH}/sub.csv", "weight": 1.13},
    {"file2": f"{SUBMISSION_FILES_PATH}/submission.csv", "weight": 1.1},
    {"file3": f"{SUBMISSION_FILES_PATH}/submission1.csv", "weight": 1},
    {"file4": f"{SUBMISSION_FILES_PATH}/submission_ridge.csv", "weight": 1.12},
]


# In[ ]:


subs = submission.copy()
for i, ensemble in enumerate(ENSEMBLES):
    print(ensemble)
    tmp = pd.read_csv(ensemble[f'file{i}'])
    subs[f"predicted_file{i}"] = tmp["Predicted"]

subs.drop(columns=["Id", "Predicted"], inplace=True)


# In[ ]:


subs.head(10)


# ## Finding Correlation between submissions

# In[ ]:


# Compute the correlation matrix
corr = subs.corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 12))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, annot=True, fmt="g",
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
ax.set_ylim(corr.shape[0], 0)
plt.yticks(rotation=0)


# ## Simple Mean Ensemble

# In[ ]:


def mean_ensemble(row):
    return np.mean(row.values)


# In[ ]:


submission["Predicted"] = subs.apply(mean_ensemble, axis=1)
submission.to_csv("submission.csv", index=False)


# In[ ]:


submission.head()

