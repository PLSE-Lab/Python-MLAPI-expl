#!/usr/bin/env python
# coding: utf-8

# ### Ensemble methods: classifiers and majority voting
# The goal of ensemble methods is to combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability / robustness over a single estimator [[1]](https://scikit-learn.org/stable/modules/ensemble.html).
# Here we shall look at an averaging method known as **majority voting**. In majority voting, the predicted class label for a particular sample is the class label that represents the majority 
# ([mode](https://en.wikipedia.org/wiki/Mode_&#40;statistics&#41;))
# of the class labels predicted by each individual classifier [[2]](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier). 
# 
# 
# 

# In[ ]:


import pandas as pd
import numpy  as np


# We shall read in the predictions, *i.e.* the `submission.csv` files,  from an odd number of estimators. For this example we shall use predictions from 
# a [logistic regression](https://www.kaggle.com/carlmcbrideellis/logistic-regression-classifier-minimalist-script), 
# a [random forest](https://www.kaggle.com/carlmcbrideellis/random-forest-classifier-minimalist-script).
# a [neural network](https://www.kaggle.com/carlmcbrideellis/very-simple-neural-network-for-classification),
# a [Gaussian process classifier](https://www.kaggle.com/carlmcbrideellis/gaussian-process-classification-sample-script),
# and finally 
# a [Support Vector Machine classifier](https://www.kaggle.com/carlmcbrideellis/support-vector-classifier-minimalist-script):

# In[ ]:


LogisticRegression        = pd.read_csv("../input/logistic-regression-classifier-minimalist-script/submission.csv")
RandomForestClassifier    = pd.read_csv("../input/random-forest-classifier-minimalist-script/submission.csv")
neural_network            = pd.read_csv("../input/very-simple-neural-network-for-classification/submission.csv")
GaussianProcessClassifier = pd.read_csv("../input/gaussian-process-classification-sample-script/submission.csv")
SupportVectorClassifier   = pd.read_csv("../input/support-vector-classifier-minimalist-script/submission.csv")      


# we shall now calculate the mode, using [pandas.DataFrame.mode](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mode.html):

# In[ ]:


all_data = [ LogisticRegression['Survived'] , 
             RandomForestClassifier['Survived'], 
             neural_network['Survived'], 
             GaussianProcessClassifier['Survived'], 
             SupportVectorClassifier['Survived'] ]

votes       = pd.concat(all_data, axis='columns')

predictions = votes.mode(axis='columns').to_numpy()


# and finally we shall produce a new `submission.csv` file whose predictions are now the mode of all of the above estimators:

# In[ ]:


output = pd.DataFrame({'PassengerId': neural_network.PassengerId, 
                       'Survived'   : predictions.flatten()})
output.to_csv('submission.csv', index=False)


# ### Related reading:
# * [Dymitr Ruta and Bogdan Gabrys "Classifier selection for majority voting", Information Fusion,
# Volume 6 Pages 63-81 (2005)](https://www.sciencedirect.com/science/article/abs/pii/S1566253504000417)
