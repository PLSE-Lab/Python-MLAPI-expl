#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:





#                                         **METERICS in LOGISTIC REGRESSION***
# 
# PRECISION:
# ----------
# Precision means the percentage of your results which are relevant. - TP/(TP+FP) . Measured for total actual with positive
# 
# RECALL :
# --------
# Recall or Sensitivity or True Positive rate refers to the percentage of total relevant results correctly classified by your algorithm - TP/(TP+FN) . Measured for total prediction with positive
# 
# 
# ACCURACY:
# ---------
# Accuracy refers to ratio of correct classification with incorrect classification
# 
# Consider in amazon site, if you search product and it able to list 20 products out of which 10 are relevant then we see Recall is 100% (10/10)  whereas precision is 50% (10/20)
# 
# So there should be trade-off between Precision and Recall to have model optimal model for some cases which is provided by F1 score which is hormonic mean of Precision and recall. for other model we can maximise either of this
# 
# SPECIFICITY:
# -----------
# Specificity or TNR (True Negative Rate): Number of items correctly identified as negative out of total negatives- TN/(TN+FP) . Measured for total prediction
# 
# False Positive Rate or Type I Error: Number of items wrongly identified as positive out of total true negatives- FP/(FP+TN). Measured for total prediction
# False Negative Rate or Type II Error: Number of items wrongly identified as negative out of total true positives- FN/(FN+TP). Measured for total prediction
# 
# ROC curve:
# ----------
# 
# It is curve used to measure relationship between true positive rate vs true negative rate.
# 
# The ROC curve is a useful tool for a few reasons:
# 
# 
# The curves of different models can be compared directly in general or for different thresholds.
# The area under the curve (AUC) can be used as a metric which falls between 0 and 1 with a higher number indicating better classification performance.
# Helps to identify threshold.
# 
# 
# 
# Reference:
# 
# https://towardsdatascience.com/precision-vs-recall-386cf9f89488
# 
# https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
# 
# https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c
# 
