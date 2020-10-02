#!/usr/bin/env python
# coding: utf-8

# This Notebook is copied from https://www.kaggle.com/osciiart/understanding-lwlrap-sorry-it-s-in-japanese
# 
# I just translated the text with google translator.

# In[ ]:


# Library loading
import numpy as np
import sklearn.metrics


# In[ ]:


# LwLRAP Calculation function
# from official code https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8#scrollTo=cRCaCIb9oguU
def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      truth: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.  Then the overall unbalanced lwlrap is
        simply np.sum(per_class_lwlrap * weight_per_class)
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    # Space to store a distinct precision value for each class on each sample.
    # Only the classes that are true for each sample will be filled in.
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = (
            _one_sample_positive_class_precisions(scores[sample_num, :],
                                                  truth[sample_num, :]))
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
            precision_at_hits)
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    # Form average of each column, i.e. all the precisions assigned to labels in
    # a particular class.
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
    #                = np.sum(per_class_lwlrap * weight_per_class)
    return per_class_lwlrap, weight_per_class


# # 1 sample Calculation per  
# If there is one correct answer label.<br>
# Assume that the class is A, B, C.<br>
# Consider the case where the correct answer label = A, prediction = (A: 0.7, B: 0.1, 0.2) as an example.<br>
# First, rank the predictions (swing numbers in order of decreasing value).<br>
# -> Forecast = (A: 1, B: 3, C: 2)<br>
# Score = 1 <br>
# Calculated as the number of correct answers up to the rank of the correct answer label / rank of the correct answer label.<br>
# in this case, Correct label rank = 1<br>
# Number of correct answers from 1 to the rank of correct label => number of correct answers from rank 1 to 1 = 1<br>
# So,<br>
# Score = 1/1 = 1.0.<br>

# In[ ]:


# Let's actually calculate.
y_true = np.array([1, 0, 0,])
y_score = np.array([0.7, 0.1, 0.2])
pos_class_indices, precision_at_hits = _one_sample_positive_class_precisions(y_score, y_true)
print("Correct answer label", pos_class_indices)
print("Score", precision_at_hits)


# Another example<br>
# If the correct answer label = A, prediction = (A: 0.1, B: 0.7, 0.2).<br>
# Ranked prediction = (A: 3, B: 1, C: 2)<br>
# Correct answer label rank = 3<br>
# Number of correct answers from 1 to the rank of correct label = number of correct answers from rank 1 to 3 = 1<br>
# So,Score = 1/3 = 0.33.<br>

# In[ ]:


# Let's actually calculate.
y_true = np.array([1, 0, 0,])
y_score = np.array([0.1, 0.7, 0.2])
pos_class_indices, precision_at_hits = _one_sample_positive_class_precisions(y_score, y_true)
print("Correct answer label", pos_class_indices)
print("Score", precision_at_hits)


# If there are multiple correct answer labels.<br>
# Assume that the class is A, B, C.<br>
# Consider the case where the correct answer label = A, C prediction = (A: 0.7, B: 0.1, 0.2) as an example.<br>
# Score is calculated for each correct answer label.<br>
# First, when calculating the score for the correct answer label A,<br>
# Ranked prediction = (A: 1, B: 3, C: 2)<br>
# Correct label rank = 1<br>
# Number of correct answers from 1 to the rank of correct label = number of correct answers from rank 1 to 1 = 1<br>
# So,Score = 1/1 = 1.0.<br>
# <br>
# Next, when calculating the score for the correct answer label C,<br>
# Ranked prediction = (A: 1, B: 3, C: 2)<br>
# Correct answer label rank = 2<br>
# Number of correct answers from 1 to the rank of correct label = number of correct answers from rank 1 to 2 = 2<br>
# So,Score = 2/2 = 1.0<br>

# In[ ]:


# Let's actually calculate.
y_true = np.array([1, 0, 1,])
y_score = np.array([0.7, 0.1, 0.2])
pos_class_indices, precision_at_hits = _one_sample_positive_class_precisions(y_score, y_true)
print("Correct answer label", pos_class_indices)
print("Score", precision_at_hits)


# Another example  <br>
# <br>
# Consider the case where the correct answer label = A, C prediction = (A: 0.1, B: 0.7, 0.2) as an example.<br>
# Score is calculated for each correct answer label.<br>
# First, when calculating the score for the correct answer label A,<br>
# Ranked prediction = (A: 3, B: 1, C: 2)<br>
# Correct answer label rank = 3<br>
# Number of correct answers from 1 to the rank of correct label = number of correct answers from rank 1 to 3 = 2<br>
# So,Score = 2/3 = 0.67.<br>
# <br>
# Next, when calculating the score for the correct answer label C,<br>
# Ranked prediction = (A: 3, B: 1, C: 2)<br>
# Correct answer label rank = 2<br>
# Number of correct answers from 1 to the rank of correct label = number of correct answers from rank 1 to 2 = 1<br>
# So,Score = 1/2 = 0.5. <br>

# In[ ]:


# Let's actually calculate.
y_true = np.array([1, 0, 1,])
y_score = np.array([0.1, 0.7, 0.2])
pos_class_indices, precision_at_hits = _one_sample_positive_class_precisions(y_score, y_true)
print("Correct answer label", pos_class_indices)
print("Score", precision_at_hits)


# Calculation for all samples<br>
# <br>
# Sample 1: Correct answer label = A, C prediction = (A: 0.1, B: 0.7, 0.2)<br>
# Sample 2: Correct answer label = B, C prediction = (A: 0.1, B: 0.7, 0.2)<br>
# Consider the case.<br>
# First, calculate the score for each class.<br>
# Score = total score for a class / number of correct labels for a class<br>
# in this case,<br>
# Sample 1 score = A: 0.6667, C: 0.5<br>
# Sample 2 score = B: 1.0, C: 1.0<br>
# So,<br>
# Class A score = 0.6667 / 1 = 0.6667<br>
# Class B score = 1.0 / 1 = 1.0<br>
# Class C score = (0.5 + 1.0) / 2 = 0.75<br>
# It becomes.<br>
# <br>
# When calculating scores for all classes, averaging the scores of each class does not take into account the bias in the number of correct labels for each class.<br>
# Frequent classes have less impact on the final score of one label,<br>
# Infrequently occurring classes have a greater effect on the final score of one label.<br>
# Therefore, a weighted average is taken with the number of occurrences of each class as a weight.<br>
# Correct label occurrence number = (A: 1. B: 1, C: 2)<br>
# Weight = Number of correct label occurrence / Total number of correct labels = (A: 1. B: 1, C: 2) / 4 = (A: 0.25, B: 0.25. C: 0.5)<br>
# Score = Sum of (score weight for each class) = A: 0.6667 0.25 + B: 1.0 0.25 + C: 0.75 0.5 = 0.7917<br>
# This is ultimately equal to the average of the scores for each label.<br>
# Average score for each label = (0.6667 + 0.5 + 1.0 + 1.0) / 4 = 0.7917 <br>

# In[ ]:


# Let's actually calculate.
y_true = np.array([[1, 0, 1,], [0, 1, 1]])
y_score = np.array([[0.1, 0.7, 0.2], [0.1, 0.7, 0.2]])
_, precision_at_hits1 = _one_sample_positive_class_precisions(y_score[0], y_true[0])
print("sample 1 Score", precision_at_hits1)
_, precision_at_hits2 = _one_sample_positive_class_precisions(y_score[1], y_true[1])
print("sample 2 Score", precision_at_hits2)
score, weight = calculate_per_class_lwlrap(y_true, y_score)
print("Each class score", score)
print("Weight of each class", weight)
LwLRAP = (score*weight).sum()
print("LwLRAP", LwLRAP)


# In[ ]:




