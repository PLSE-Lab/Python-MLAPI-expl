#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Comparing errors made by different models while predicting on same data

# ## Introduction

# Suppose you are working on a binary classification problem. You trained 3 different models on your data: say SVM,XGBoost and a Multi-Layer Perceptron (MLP). Using a command like `model.predict(x_val)`, you can get predictions made by a each of the models for your validation samples:
# 
# ```python
# svm_preds = svm.predict(x_val)
# xgb_preds = xgb.predict(x_val)
# mlp_preds = mlp.predict(x_val)
# ```
# 
# Also, you have access to the true labels for the validation data: `y_val`.
# 
# **You want to understand how the errors made by the different models compare to each other. Do the models make mistakes on the same samples? Or does each model make mistakes on different samples?**  
# 
# To answer this question, we'll use a simple but beautiful plot. But first, we need some data.
# 
# You can write the errors made by the different models to a DataFrame, like this:
# ```python
# import pandas as pd
# #Create a dictionary, keys are models, values are the predictions.
# predictions = {'True': y_val, 
#      'svm_preds': svm_preds,
#      'xgb_preds': xgb_preds,
#      'mlp_preds': mlp_preds}
# #Create df
# predictions = pd.DataFrame.from_dict(d)
# 
# #Write df to csv
# predictions.to_csv('error_analysis.csv')
# ```
# 
# You can easily create the above DataFrame using your own data and models. 
# 
# 
# For the purpose of illustrarion, I've included an `error_analysis.csv` file, which we'll use to see how we can visualize the errors made by different models.
# 
# 
# 

# ## Import data

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


predictions = pd.read_csv("../input/compare-errors-made-by-different-models/error_analysis.csv")
predictions.head()


# From above, we see that the `predictions` df contains the true label as well as the predictions made by 3 different models for each given sample. 
# 
# Let's visualize `predictions` using the `heatmap` function from `sns`.

# ## The importance of sorting

# In[ ]:


sns.heatmap(predictions, cmap = "pink", cbar=False,  yticklabels=False)
plt.title('Comparison of Predictions made by 3 models')
plt.show()


# The above plot is not very helpful. but if we first sort `predictions`, we can get a much more expressive plot:

# In[ ]:


#sort predictions
predictions = predictions.sort_values(by=['True Label', 'SVM_prediction', 'XGBoost_prediction', 'MLP_prediction' ])
predictions.head()


# In[ ]:


#make heatmap after sorting
sns.heatmap(predictions, cmap = "pink", cbar=False,  yticklabels=False)
plt.title('Comparison of Predictions made by 3 models')
plt.show()


# This heatmap is much better than the last one. *Black* is for `0`s and *white* is for 1s.
# 
# For most of the samples, all rows are either completely black or completely white, which means we're making **correct predictions**. The models seem to be doing pretty well!
# 
# But for some rows in the middle, we see that the `True label` and model predictions have different colors. These are the **misclassified samples**. Let's take a closer look at the samples which are misclassified by at least one of the models (unmatching colored ones in the middle).

# ## Samples misclassified by at least one model

# In[ ]:


#Keep only those samples for which at least one of the models makes an incorrect prediction
errors = predictions.loc[(predictions['True Label'] != predictions['SVM_prediction']) | 
                         (predictions['True Label'] != predictions['XGBoost_prediction']) | 
                         (predictions['True Label'] != predictions['MLP_prediction'])]


# In[ ]:


sns.heatmap(errors, cmap = "pink", cbar=False,  yticklabels=False)
plt.title('Incorrect predictions comparison for 3 models')
plt.show()


# This plot tells us several things:
# 1. Different models misclassify different samples. For example, the first block of samples (*black* for first 3 columns, *white* for MLP) is classified correctly by SVM and XGB, but misclassified by MLP.
# 2. There are some samples for which all three models misclassify. See the blocks in the middle where true label is *black* (0), but all 3 models predict *white* (1), and vice versa. It's interesting to note that there are a lot more samples with true label *black* (0) which are misclassified by all 3 models, as compared to samples with true label *white* (1) which are misclassifed by all 3 models. Does this mean classifying samples from class *black* (0) in inherently harder than classifying samples from class *white* (1)?

# I hope this helps you for your own analysis!
# 
# If you have suggestions for improvement, let me know!
