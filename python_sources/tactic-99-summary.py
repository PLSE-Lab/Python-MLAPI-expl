#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The aim of this notebook is to analyse the results of each of the different tactics tested.
# 
# There are a huge collection of notebooks, each one with a tactic or part of a tactic.
# The objective of each tactic is to study, analyse, and get conclusions of an very specific concept.
# 
# - [Tactic 00. Baseline](https://www.kaggle.com/juanmah/tactic-00-baseline)
# - [Tactic 01. Test classifiers](https://www.kaggle.com/juanmah/tactic-01-test-classifiers)
# - [Tactic 02. Stack classifiers](https://www.kaggle.com/juanmah/tactic-02-stack-classifiers)
# - [Tactic 03. Hyperparameter optimization](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization):
#  - [Tactic 03. Hyperparameter optimization. LR](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization-lr)
#  - [Tactic 03. Hyperparameter optimization. LDA](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization-lda)
#  - [Tactic 03. Hyperparameter optimization. KNN](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization-knn)
#  - [Tactic 03. Hyperparameter optimization. GNB](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization-gnb)
#  - [Tactic 03. Hyperparameter optimization. SVC](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization-svc)
#  - [Tactic 03. Hyperparameter optimization. Bagging](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization-bagging)
#  - [Tactic 03. Hyperparameter optimization. Xtra-trees](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization-xtra-trees)
#  - [Tactic 03. Hyperparameter optimization. Adaboost](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization-adaboost)
#  - [Tactic 03. Hyperparameter optimization. GB](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization-gb)
#  - [Tactic 03. Hyperparameter optimization. LightGBM](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization-lightgbm)
#  - [Tactic 03. Hyperparameter optimization. XGBoost](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization-xgboost)
# - [Tactic 04. Stack optimized classifiers](https://www.kaggle.com/juanmah/tactic-04-stack-optimized-classifiers)
# - [Tactic 05. Class weight](https://www.kaggle.com/juanmah/tactic-05-class-weight)
# - [Tactic 06. Feature engineering](https://www.kaggle.com/juanmah/tactic-06-feature-engineering)
# - [Tactic 07. Outlier detection](https://www.kaggle.com/juanmah/tactic-07-outlier-detection)
# - [Tactic 08. Feature selection: optimal importance](https://www.kaggle.com/juanmah/tactic-08-feature-selection-optimal-importance)
# - [Tactic 09. Feature selection: backward elimination](https://www.kaggle.com/juanmah/tactic-09-feature-selection-backward-elimination)
# - [Tactic 10. Feature selection: forward selection](https://www.kaggle.com/juanmah/tactic-10-feature-selection-forward-selection)
# 
# The result table is the collection of all the individual results in each tactic.
# 
# There are two main numbers in this table: result and score.
# **Result** is the internal score (i.e. the score on the train set),
# and **score** is the public score returned by Kaggle (i.e. the score in the test set).

# In[ ]:


import pandas as pd
import numpy as np

def compare(results, baseline_tactic, current_tactic):
    """This function compares the results between a current tactic and a baseline.
    For each result in the current tactic it compares the result and the score.

    :param results: DataFrame with results.
    :param baseline_tactic: Tactic that will be used as a baseline.
    :param current_tactic: Tactic with the results to compare.
    :return: DataFrame with model, and differences in result and score between current and baseline tactics.
    """
    comparison = pd.DataFrame(columns=['Model',
                                       'Result',
                                       'Score'])

    for row in results.query('Tactic == ' + str(current_tactic)).itertuples(index=False):
        previous = results.query('Tactic == ' + str(baseline_tactic) + ' and Model == "' + row.Model + '"')
        comparison = comparison.append({
            'Model': row.Model,
            'Result': '{:.2%}'.format((row.Result - float(previous.Result)) / float(previous.Result)),
            'Score': '{:.2%}'.format((row.Score - float(previous.Score)) / float(previous.Score))
        }, ignore_index=True)

    return comparison

results = pd.read_csv('../input/tactic-98-results/results.csv', index_col='Id', engine='python')


# # Tactic 03. Hyperparameter optimization

# In[ ]:


compare(results, 1, 3)


# The hyperparameter optimization is a MUST to get a good score.
# All models can be enhaced by optimizacion.
# 
# Models as `lg`, `bg`, `lr` and `gb` are good enough with the default parameters.
# The rest can be improved a lot.
# 
# This is valid for this data.
# For other data, probably, the results could be differnt.

# # Tactic 05. Class weight

# In[ ]:


compare(results, 3, 5)


# The classes are balanced.
# There is no need to weight them.
# 
# If the classes in the train set had the save proportion as in the test set,
# the score would be pretty bad.
# 
# Then, it's important to have the classes balanced.

# # Tactic 06. Feature engineering

# In[ ]:


compare(results, 3, 6)


# Creating more features from the original ones,
# makes generally the models to have more score.
# 
# Two models: `knn` and `lr` have a worse score with more features. 

# # Tactic 07. Outlier detection

# In[ ]:


compare(results, 6, 7)


# Isolation forest is applied to detect outliers.
# 
# Removing samples that are pointed as outliers don't improve the score.
# The more samples are removed the worst the score.

# # Tactic 08. Feature selection: optimal importance

# In[ ]:


compare(results, 6, 8)


# Only optimal importance has a tiny improve in the score.
# 
# From 70 features of the tactic 06 model, the optimal importance has 59.
# Only 11 features drop.
# 
# Some features, when added to the model, makes the score increase or decrease.
# If only the features that makes the score increase are taken,
# then the score is practically the same,
# with only 39 features.

# # Tactic 09. Feature selection: backward elimination

# In[ ]:


compare(results, 6, 9)


# Backward elimination has a very close score compared with the original using only a few less features: 61.

# # Tactic 10. Feature selection: forward selection

# In[ ]:


compare(results, 6, 10)


# Forward selection has a similar, but lower score, with only 36 features.
