#!/usr/bin/env python
# coding: utf-8

# # About this kernel
# 
# This is to show how different ways of measuring the performance of this model will yield very different results, and as a conclusion we will see that QWK is perhaps far from the best metrics for this type of data. We will show
# 
# - Confusion Matrix: How "blurred" the diagonal line is, compared to a perfect scoring.
# - Classification Report: How the f1 score decrease dramatically as we go into more rare classes.
# - LWK vs QWK: How a simple weighting changes the score by 10%
# - Prediction Histogram: How the predicted classes frequency are pretty "off" compared to ground truth.

# In[ ]:


import os
import math

import numpy as np
import pandas as pd
from sklearn import metrics
import plotly.graph_objs as go
import plotly.express as px


# # Reproducing simple baseline
# 
# Essentially [this kernel](https://www.kaggle.com/suicaokhoailang/an-embarrassingly-simple-baseline-0-960-lb), but applied on training set.

# In[ ]:


df = pd.read_csv("../input/liverpool-ion-switching/train.csv")
train = df.copy()

n_groups = df.shape[0] // 50000
df["group"] = 0
for i in range(n_groups):
    ids = np.arange(i*50000, (i+1)*50000)
    df.loc[ids,"group"] = i

for i in range(n_groups):
    sub = df[df.group == i]
    signals = sub.signal.values
    imax, imin = math.floor(np.max(signals)), math.ceil(np.min(signals))
    signals = (signals - np.min(signals))/(np.max(signals) - np.min(signals))
    signals = signals*(imax-imin)
    df.loc[sub.index,"pred_open_channels"] = np.array(signals,np.int)

y_true = df.open_channels.values
y_pred = df.pred_open_channels.values


# # Visualize normalization confusion matrix
# 
# We will normalize along the true axis.

# In[ ]:


get_ipython().run_cell_magic('time', '', "cm = metrics.confusion_matrix(y_true, y_pred, normalize='true')")


# In[ ]:


fig = px.imshow(cm)
fig.show()


# # Looking at f1 score by class

# In[ ]:


get_ipython().run_line_magic('time', 'report = metrics.classification_report(y_true, y_pred)')


# In[ ]:


print(report)


# # Comparing various kappa scoring

# In[ ]:


get_ipython().run_cell_magic('time', '', 'lwk = metrics.cohen_kappa_score(y_true, y_pred, weights=\'linear\')\nqwk = metrics.cohen_kappa_score(y_true, y_pred, weights=\'quadratic\')\\\n\nprint("Linear Weighted Kappa Score:", lwk)\nprint("Quadratic Weighted Kappa Score:", qwk)')


# # Looking at prediction histogram

# In[ ]:


true_bins = np.bincount(y_true)
pred_bins = np.bincount(y_pred.astype(int))[:10]


# In[ ]:


fig = go.Figure([
    go.Bar(y=true_bins, name='True Labels'),
    go.Bar(y=pred_bins, name='Pred Labels')
])

fig.show()

