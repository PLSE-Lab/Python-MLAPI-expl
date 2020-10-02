#!/usr/bin/env python
# coding: utf-8

# ### F1 Score:
# 
# F1 Score is harmonic mean of Precision and Recall.
# 
# ![F1 Score](https://svgshare.com/i/M7d.svg)
# 

# ## Precision: 
# It tells you how weak is your model in predicting positive Classes. If number of False Positives are more than True Positives then your model has less precision.
# 
# ## Recall:
# It tells you how weak is your model in covering True Classes. If number of False Negatives are more than True Positives then your model has less recall.
# 
# For more info on Precision and Recall, Please go through: https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2

# *  F1-Score can be easily calculated for a binary classification Problem.
# *  What if we have multi-class classification ? How can we calculate F1-Score and if calculated for each label how can we merge / consolidate them ?

# Let's try to address these questions by considering an Example:
# 
# We have a dataset of 3 Labels Cat, Dog and Fish. We have built a multi-class classifier that predicts one label among these 3 labels and let's consider below as the the outputs for each Label. (One vs All)
# 
# 

# The above values can be represented as below confusion Matrix:
# 
# ![Confusion Matrix](https://i.ibb.co/swZrksj/Screenshot-from-2020-06-17-14-13-01.png)
# 
# From the above table we can deduce the following for each Label:
# 
# Cat: 10 TP; 15 FP; 13 FN
# 
# Dog: 20 TP; 11 FP; 12 FN
# 
# Fish: 13 TP; 12 FP; 13 FN

# Now F-1 Score for each label can be calculated by calculating individual Precisions and Recalls by
# 
# Precision = T.P / (T.P + F.P)
# Recall = T.P / (T.P + F.N)
# 
# Ex: For Cat we have the following:
# Precision = 10 / (10+15) = 0.4
# Recall = 10 / (10+13) = 0.43
# 
# Therefore F1-Score for ** Cat is 0.414 **
# 
# Similarly F1-Scores for other labels are ** Dog 0.635, Fish 0.51 **

# We acheived F1 Scores for independent Labels. How can we aggregate these F1-Scores into a single F1-Score for evaluating the Model ?

# We have two type of Aggregations:
# 
# **Macro Averaged F1-Score and Micro-Averaged F1-Score**

# ### Macro Averaged F1-Score:
# Here we simple average all the F1-Scores and calculate a mean F1-Score.
# 
# Average of all the F1-Scores result in **0.52**
# 
# But simply weighing all the F1-Scores isn't a fair way to estimate a model perfomance. Because this metric doesn't perform good on imbalanced dataset. A good explaination on the same is provided below:
# 
# https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin

# ## Micro Averaged F1-Score:
# 
# Here instead of calculating each label's F1-Score, we derive the F1-Score by calculating Precision and Recall by summing all the TPs and Type Errors instead of calculating for each Label:
# 
# Total T.Ps = 10+20+13 = 43
# Total F.Ps = 15+11+12 = 38
# Total F.Ns = 13+12+13 = 39
# 
# Precision = 43/(43+38) = 0.53
# Recall = 0.524
# 
# F1-Score = 0.526
# 
# 

# In[ ]:




