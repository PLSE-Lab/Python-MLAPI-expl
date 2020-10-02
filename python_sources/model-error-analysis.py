#!/usr/bin/env python
# coding: utf-8

# # Model Error analysis
# In this kernel I did some basic error analysis of one of my models predictions.  
# This was a densenet model with lb near 0.96 but the types of error will be same  
# for most of the models.
# 
# I hope someone benifits from this.

# imports and Loading validation set targets and predictions

# In[ ]:


import os
import gc
import cv2
import random
import sklearn


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

plt.style.use('bmh')

targets0 = np.load("../input/train-single-best-model-error-analysis/targets0.npy")
targets1 = np.load("../input/train-single-best-model-error-analysis/targets1.npy")
targets2 = np.load("../input/train-single-best-model-error-analysis/targets2.npy")
preds0 = np.load("../input/train-single-best-model-error-analysis/preds0.npy")
preds1 = np.load("../input/train-single-best-model-error-analysis/preds1.npy")
preds2 = np.load("../input/train-single-best-model-error-analysis/preds2.npy")


# helper function

# In[ ]:


def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    sns.set(font_scale=1.5)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


# # 1) vowel_diacritic

# ## Confusion matix

# In[ ]:


confusion_matrix = sklearn.metrics.confusion_matrix(np.array(targets1), np.array(preds1))
print_confusion_matrix(confusion_matrix, [i for i in range(confusion_matrix.shape[0])], figsize = (20,15), fontsize=20)
plt.show()


# ## Error rate per class

# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(targets1, preds1))


# In[ ]:


score_dict = classification_report(targets1, preds1, output_dict=True)
support, recalls, precision, f1 , cls = [], [], [], [], []
for i in range(11):
    cls.append(i)
    support.append(score_dict[str(i)]['support'])
    recalls.append(score_dict[str(i)]['recall'])
    precision.append(score_dict[str(i)]['precision'])
    f1.append(score_dict[str(i)]['f1-score'])

f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 12))
ax1.bar(cls, recalls, width=0.8, bottom=None, align='center', color='#F65058')
ax1.set_title("recall",fontsize= 25)
ax2.bar(cls, precision, width=0.8, bottom=None, align='center', color='#FBDE44')
ax2.set_title("precision",fontsize= 25)
ax3.bar(cls, support, width=0.8, bottom=None, align='center', color="#28334A")
ax3.set_title("support",fontsize= 25)
plt.tight_layout()
plt.show()


# # 2) consonant_diacritic

# ## Confusion matrix

# In[ ]:


confusion_matrix = sklearn.metrics.confusion_matrix(np.array(targets2), np.array(preds2))
print_confusion_matrix(confusion_matrix, [i for i in range(confusion_matrix.shape[0])], figsize = (20,15), fontsize=20)
plt.show()


# ## Error rate per class

# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(targets2, preds2))


# In[ ]:


score_dict = classification_report(targets2, preds2, output_dict=True)
support, recalls, precision, f1 , cls = [], [], [], [], []
for i in range(7):
    cls.append(i)
    support.append(score_dict[str(i)]['support'])
    recalls.append(score_dict[str(i)]['recall'])
    precision.append(score_dict[str(i)]['precision'])
    f1.append(score_dict[str(i)]['f1-score'])

f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 12))
ax1.bar(cls, recalls, width=0.8, bottom=None, align='center', color='#F65058')
ax1.set_title("recall",fontsize= 25)
ax2.bar(cls, precision, width=0.8, bottom=None, align='center', color='#FBDE44')
ax2.set_title("precision",fontsize= 25)
ax3.bar(cls, support, width=0.8, bottom=None, align='center', color="#28334A")
ax3.set_title("support",fontsize= 25)
plt.tight_layout()
plt.show()


# # 3) grapheme_root

# ## Error rate per class

# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(targets0, preds0))


# In[ ]:


score_dict = classification_report(targets0, preds0, output_dict=True)
support, recalls, precision, f1 , cls = [], [], [], [], []
for i in range(168):
    cls.append(i)
    support.append(score_dict[str(i)]['support'])
    recalls.append(score_dict[str(i)]['recall'])
    precision.append(score_dict[str(i)]['precision'])
    f1.append(score_dict[str(i)]['f1-score'])

f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 12))
ax1.bar(cls, recalls, width=0.8, bottom=None, align='center', color='#F65058')
ax1.set_title("recall",fontsize= 25)
ax2.bar(cls, precision, width=0.8, bottom=None, align='center', color='#FBDE44')
ax2.set_title("precision",fontsize= 25)
ax3.bar(cls, support, width=0.8, bottom=None, align='center', color="#28334A")
ax3.set_title("support",fontsize= 25)
plt.tight_layout()
plt.show()


# In[ ]:


sup = [i/(1 * max(support)) for i in support]
f, ax = plt.subplots(figsize=(20, 5))
ax.bar(cls, recalls, width=0.7, bottom=None, align='center', color='#F65058')
ax.bar(cls, sup, width=0.7, bottom=None, align='center', color="#28334A")
ax.set_title("Recall (red), with support (black)",fontsize= 25)
plt.show()


# ### We can see that the classes with least recalls have less support

# ## confusion matrix

# ### Open image in new tab

# In[ ]:


confusion_matrix = sklearn.metrics.confusion_matrix(np.array(targets0), np.array(preds0))
print_confusion_matrix(confusion_matrix, [i for i in range(confusion_matrix.shape[0])], figsize = (50,40), fontsize=6)
plt.show()


# In[ ]:


confusion_matrix = sklearn.metrics.confusion_matrix(np.array(targets0), np.array(preds0))
errors = []
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[0]):
        if confusion_matrix[i][j] != 0 and i != j:
            errors.append(confusion_matrix[i][j])
fig, ax = plt.subplots(figsize=(20,5))
sns.countplot(ax=ax, x=errors)
ax.set_title("Count of Non diagonal non zero entries",fontsize= 25)
plt.show()


# ### Non diagonal entries which are greater than 8 

# In[ ]:


confusion_matrix = sklearn.metrics.confusion_matrix(np.array(targets0), np.array(preds0))
print("True\tPred\tcount/support")
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[0]):
        if confusion_matrix[i][j] != 0 and i != j:
            if confusion_matrix[i][j] > 8:
                print(f"{i}\t{j}\t{confusion_matrix[i][j]}/{support[i]}")


# ## Please Upvote if you found this useful
