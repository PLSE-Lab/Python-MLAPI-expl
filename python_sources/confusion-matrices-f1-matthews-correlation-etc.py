#!/usr/bin/env python
# coding: utf-8

# # Confusion matrices, F1, Matthews Correlation, etc
# 
# In this notebook I discuss three simple and relatively closely related concepts:
# * Confusion matrix
# * Precision and recall
# * F2 score
# * Matthews correlation
# 
# The confusion matrix is a fundamental concept in measuring machine learning classifier performance. The remaining items, meanwhile, are related measurements, all derived from the confusion matrix, which attempt to encode some aspect of the matrix as a single number. They each have benefits and tradeoffs, but all are in wide enough use that as a data scientist you should probably know them!
# 
# ## Confusion matrix
# The confusion matrix contains the list of classes on either axis and numbers in the middle. In most depictions, the top is the ground truth and the left axis is the classification predicted by the classifier: $y$ and $y_{pred}$, respectively. The diagonal entries in the matrix are good---they correspond with good classifications. The off-diagonal entries are bad; they correspond with classification errors by the model.
# 
# What makes interpreting a confusion matrix valuable is the fact that it gives you a strong understanding of what types of errors your model made. Regardless of the classification task, certain types of errors are likely to be more likely than others, indicating great confusion about (and less separability between) the two classes in question (at least with respect to other classes).
# 
# For example:
# 
# ```
#     0    1
# 0 250  5
# 1 25   250
# ```
# 
# This report shows that the classifier rarely mistakes class 1 for class 0, but often mistakes class 0 for class 1. This is an asymmetric error and it's valuable knowledge that we can use the tune the model.
# 
# ## Precision and recall
# The confusion matrix is an easily interpretable and valuable metric, but it has two problems: it gives numerical values which can be hard to interpret with respect to one another, and when there ar a lot of classes there are a lot of pairwise relationships and it can be hard to ingest them all. For these reasons, and the genuine usefulness of the confusion matrix, there are a ton of metrics that depend on elements on the matrix for interpretation.
# 
# Precision and recall are among these metrics. These are per-class attributes which measure specific desirable attributes of model performance.
# 
# For a given class C. **Precision** is the percentage of records classified C which are actually C, e.g. how "precise" the model is. Meanwhile, **recall** is the percentage of _all_ records of class C classified as such. They have precise formulas which are based on per-class true positive, false positive, true negative, and false negative rates.
# 
# Precision is a valuable metric when it is important that hits really be hits, e.g. the true positive rate is what matters most. For example, a web search engine.
# 
# Recall is a valuable metric when it is important that as many relevant entries as possible are returned, e.g. keeping the false negative rate low is what matters most. For example, cancer screenings.
# 
# The trade-off therein is in the fact that a low precision naturally corresponds with a high recall, and vice versa. You cannot be precise without leaving a few records out, and you cannot have high recall without misclassifying a few things. The relative rate of these values on a set of classes tells you a lot about the types of errors your model is making.
# 
# For example, here is a `sklearn.classification_report` from a real model I built:
# 
# ```
# precision    recall  f1-score   support
# 
#            0       0.90      0.59      0.71      1399
#            1       0.64      0.92      0.75      1109
# 
#    micro avg       0.73      0.73      0.73      2508
#    macro avg       0.77      0.75      0.73      2508
# weighted avg       0.78      0.73      0.73      2508
# ```
# 
# For my notes:
# 
# > It's interesting to note from the classification report above that our model is under-confident when classifying hamburgers (class 0) but over-confident when classifying hamburgers (class 1). 90% of hamburgers classified as such are actually hamburgers, but only 59% of all actual hamburgers are classified correctly. On the other hand, just 64% of sandwiches classified as such are actually sandwiches, but 92% of sandwiches are classified correctly.
# 
# ## F1 score
# Sometimes having two numbers per class to look at is still too much, and you want to pick just one.
# 
# The F1 score is the harmonic mean of precision and recall, which weighs each of the two equally. Its best possible value is 1 and its worst possible value is 0. One could also define and use a wieghted F1 score, if different classes have different levels of importance to your model.
# 
# ## Matthews correlation
# Sometimes have claswise information is too much, and you just want one number to summarize all of that information. That is essentially a metric&mdash;something that you can optimize. It is also a correlation&mdash;a specific type of metric which returns a score of between -1 (indicating low information and high entropy) and 1 (indicating perfect information and no entropy). For more on metrics see my notebook ["Classification metrics with Seattle rain"](https://www.kaggle.com/residentmario/classification-metrics-with-seattle-rain/). For more on correlations see my notebooks on the two most common correlation scores (the Matthews correlation can be considered a third most common): [Pearson's r](https://www.kaggle.com/residentmario/pearson-s-r-with-health-searches) and [Spearman correlation](https://www.kaggle.com/residentmario/spearman-correlation-with-montreal-bikes/).
# 
# Amongst the possible choices of classification metrics, the one which is considered in the literature to be the best summary of the confusion matrix (if that is what you are looking for) is **Matthews correlation**. MCC, as is sometimes called, can be stated as the following formulation of positive and negative rates:
# 
# $$MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$
# 
# The meaning of the Matthews correlation is difficult to state in simple terms.
# 
# The Matthews correlation is considered an excellent summary of the model confusion matrix because it is much less sensitive to differences in the size of the classes than more naive metrics are, like e.g. accuracy (in which accuracy for the most common type of thing will dominate).
