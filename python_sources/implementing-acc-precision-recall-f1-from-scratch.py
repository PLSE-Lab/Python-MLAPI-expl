#!/usr/bin/env python
# coding: utf-8

# ### Assume Postive class : 1 and Negative class : 0

# In[ ]:


ground_truth = [1,0,1,1,1,0,1,0,1,1]
prediction   = [1,1,1,0,1,0,1,1,1,0]


# ### True Positive:
# If model predicts Positive class correctly then its True Positive.

# In[ ]:


#True Positive
def true_positive(ground_truth, prediction):
    tp = 0
    for gt, pred in zip(ground_truth, prediction):
        if gt == 1 and pred == 1:
            tp +=1
    return tp


# ### True Negative:
# If model predicts Negative class correctly then its True Negative.

# In[ ]:


#True Negative
def true_negative(ground_truth, prediction):
    tn = 0
    for gt, pred in zip(ground_truth, prediction):
        if gt == 0 and pred == 0:
            tn +=1
    return tn


# ### False Positive:
# If model predicts Positive class incorrectly then its False Positive.

# In[ ]:


#False Positive
def false_positive(ground_truth, prediction):
    fp = 0
    for gt, pred in zip(ground_truth, prediction):
        if gt == 0 and pred == 1:
            fp +=1
    return fp


# ### False Negative:
# If model predicts Negative class incorrectly then its False Negative.

# In[ ]:


#False Negative
def false_negative(ground_truth, prediction):
    fn = 0
    for gt, pred in zip(ground_truth, prediction):
        if gt == 1 and pred == 0:
            fn +=1
    return fn


# In[ ]:


true_positive(ground_truth, prediction)


# In[ ]:


true_negative(ground_truth, prediction)


# In[ ]:


false_positive(ground_truth, prediction)


# In[ ]:


false_negative(ground_truth, prediction)


# ### Accuracy Score = (TP + TN)/ (TP + TN + FP + FN) 

# In[ ]:


def accuracy(ground_truth, prediction):
    tp = true_positive(ground_truth, prediction)  
    fp = false_positive(ground_truth, prediction)  
    fn = false_negative(ground_truth, prediction)  
    tn = true_negative(ground_truth, prediction)  
    acc_score = (tp + tn)/ (tp + tn + fp + fn)  
    return acc_score


# In[ ]:


accuracy(ground_truth, prediction)


# In[ ]:


### Lets comapre with Sklearn accuracy_score
from sklearn import metrics
metrics.accuracy_score(ground_truth, prediction)


# ### Precision = TP/ (TP + FP) 

# In[ ]:


def precision(ground_truth, prediction):
    tp = true_positive(ground_truth, prediction)  
    fp = false_positive(ground_truth, prediction)  
    prec = tp/ (tp + fp)  
    return prec


# In[ ]:


precision(ground_truth, prediction)


# In[ ]:


### Lets comapre with Sklearn precision
from sklearn import metrics
metrics.precision_score(ground_truth, prediction)


# ### Recall = TP/ (TP + FN) 

# In[ ]:


def recall(ground_truth, prediction):
    tp = true_positive(ground_truth, prediction)  
    fn = false_negative(ground_truth, prediction)  
    prec = tp/ (tp + fn)  
    return prec


# In[ ]:


recall(ground_truth, prediction)


# In[ ]:


### Lets comapre with Sklearn precision
from sklearn import metrics
metrics.recall_score(ground_truth, prediction)


# ### For a good model, Precision and Recall values should be high.

# ### F1 = 2PR/ (P + R) 
# 
# similarly,
# 
# ### F1 = 2TP/ (2TP + FP + FN)  

# In[ ]:


def f1(ground_truth, prediction):
    p = precision(ground_truth, prediction)
    r = recall(ground_truth, prediction)
    f1_score = 2 * p * r/ (p + r) 
    return f1_score


# In[ ]:


f1(ground_truth, prediction)


# In[ ]:


### Lets comapre with Sklearn precision
from sklearn import metrics
metrics.f1_score(ground_truth, prediction)

