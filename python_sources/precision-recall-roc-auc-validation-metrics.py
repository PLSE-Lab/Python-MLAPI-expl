#!/usr/bin/env python
# coding: utf-8

# ## Objective
# The objective of this kernel is to walk through the validation metrices to consider for an imbalanced classification problem.
# Intent is to go deeper into the following metrices,
# - **Accuracy**
# - **Precision** 
# - **Recall** 
# - **True Positive Rate**
# - **False Positive Rate**
# - **Receiver Operating Curve - ROC**
# - **Precision Recall Curve - PRC**
# - **F1 Score**
# - **Confusion Matrix**
# 
# ### Terminologies
# We shall come across the following terminologies while getting deeper in to the above said metrices are calculated
# 1. True Positives(**TP**), False Positives(**FP**), True Negatives(**TN**), False Negatives(**FN**)
# 2. Binary Cross Entropy
# 3. Sparse Categorical Cross Entropy
# 
# I found it is hard to remember the definition of these terms and often bumped into confusion
# 
# 
# ## Approach
# Approach is to introduce the above said concepts and terms by doing step by step on digit recognition problem. Since the digit recognizer has 10 classes that is 0-9 digits of numeric system, We shall simplify the classes into 2(Recognized, Not Recognized) eventually.
# To achieve this, we shall split the dataset into train, validation and test sets. Here we shall calculate the metrices on the test set.
# **Since we are focusing on the concepts, let us ignore the actual test data because we are not aware what class they belong to.**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split


import os
print(os.listdir("../input"))

train_df = pd.read_csv('../input/train.csv')

y_train = train_df.label.values
X_train = train_df.drop(columns=["label"]).values
X_train = X_train / 255.0

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.4, random_state=42)
print(X_train.shape, X_test.shape)
print(train_df.shape)


# In[ ]:


import tensorflow as tf
print(tf.__version__)

from sklearn.metrics import accuracy_score


# ## Shallow Net
# Let us build an extremely shallow network with no hidden layers or convolution to just get some predictions on the dataset.

# In[ ]:


def incomplete_model(X, y, X_test, epochs=1):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(784,)),
        #tf.keras.layers.Dense(256, activation=tf.nn.relu),
        #tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=epochs, verbose=0)
    predictions = model.predict(X_test)
    return np.argmax(predictions, axis=1)

classifications = incomplete_model(X_train, y_train, X_test, 2) 


# ## Accuracy
# To scribe in words, "How good is the classification model?"
# 
# $$\frac{True \ Positives + True \ Negatives}{Total \ No. \ of \ Samples}$$

# In[ ]:


binary_classifications = np.absolute(y_test - classifications)
binary_classifications = np.clip(binary_classifications, 0, 1)
accuracy = 1 - (np.sum(binary_classifications) / y_test.shape[0])
print(accuracy)


# ## Transform Multi-Class to Binary Classification
# For clear illustration of the topics of interest, we shall transform this multi-class classification into binary classification problem. ie Our end goal of this section is to make digit recognizer into TRUE or FALSE problem.
# - Take a sample of 3000 observations from our test set as our universe
# - Consider predicting only the digit 9 from the universe.
# - Since 9 alone is our relevant data, a multi class problem simplified to a binary class one

# In[ ]:


NUM_TO_ANALYSE = 9
SAMPLE_COUNT = 3000

sample_indices = np.random.randint(low=0, high=y_test.shape[0], size=SAMPLE_COUNT)

actual_sample = y_test[sample_indices]
predicted_sample = classifications[sample_indices]


# ## Confusion Matrix
# 
# A confusion matrix is used to descriribe the performance of a binary classification model. There are four basic terms to be pondered
# 
# - **True Positives**: These are the samples predicted correctly
# - **True Negatives**: Predicted as **FALSE** but they are actually **FALSE**
# - **False Positives or Type 1 Error**: Predicted as **TRUE** but they are actually **FALSE**
# - **False Negatives or Type 2 Error**: Predicted as **FALSE** but they are actually **TRUE**
# 
# |   	|   Actual Positive	|   Actual Negative	| 
# |---	|---	|---	|
# |Predicted Positive	|   True Positives	|   False Positives	|
# |Predicted Negative	|   False Negatives	|   True Negatives	|

# In[ ]:


def confusion_matrix(predicted, actual, klass=NUM_TO_ANALYSE):
    
    actual_indices = np.where(actual == klass)[0]   
    predicted_indices = np.where(predicted == klass)[0]    
    not_in_predicted_indices = np.where(predicted != klass)[0]    
    not_in_actual_indices = np.where(actual != klass)[0]
    
    # True Positives: TPs are count of rightly predicted 
    true_positive_indices = np.where(actual[predicted_indices] == klass)[0]
    TRUE_POSITIVES = len(true_positive_indices)
    
    # False Positives: Failed to predict correctly
    false_positive_indices = np.where(actual[predicted_indices] != klass)[0]
    FALSE_POSITIVES = len(false_positive_indices)
    
    # True Negatives: Predicted as not part of the class and they are true in the actuals
    true_negative_indices = predicted[not_in_actual_indices]
    TRUE_NEGATIVES = len(np.where(true_negative_indices != klass)[0])
    
    # False Negatives: False negatives are not predicted as the class of interest but they are actually belongs to the class
    false_negative_indices = actual[not_in_predicted_indices]
    FALSE_NEGATIVES = len(np.where(false_negative_indices == klass)[0])
    
    return {'TP': TRUE_POSITIVES, 'FP': FALSE_POSITIVES, 'TN': TRUE_NEGATIVES, 'FN': FALSE_NEGATIVES}


metrices = confusion_matrix(predicted_sample, actual_sample, NUM_TO_ANALYSE)

print(metrices['TP'], metrices['FP'], metrices['TN'], metrices['FN'])
print(sum([metrices['TP'], metrices['FP'], metrices['TN'], metrices['FN']]))


# In[ ]:


class Metrics:
    metrics = None
    precision = None
    recall = None
    f1_score = None
    tpr = None
    fpr = None
    
    def __init__(self, metrics):
        self.metrics = metrics
        
    def calculate(self):
        self.precision = self.metrics['TP'] / (self.metrics['TP'] + self.metrics['FP'])
        self.recall = self.metrics['TP'] / (self.metrics['TP'] + self.metrics['FN'])
        self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)
        self.tpr = self.metrics['TP'] / (self.metrics['TP'] + self.metrics['FN'])
        self.fpr = self.metrics['FP'] / (self.metrics['FP'] + self.metrics['TN'])
        
metrics = Metrics(metrices)
metrics.calculate()
print(metrics.precision)


# ## Precision
# To scribe in words, "When it predicts TRUE, how often the model is correct"
# 
# $$\frac{True \ Positives}{True \ Positives + False \ Positives}$$

# In[ ]:


def precision(metrics):
    return metrics['TP'] / (metrics['TP'] + metrics['FP'])

metrices['precision'] = precision(metrices)
metrices['precision']


# ## Recall(or Sensitivity/TPR)
# To scribe in words, TPR is "If it is TRUE, how often the model predicts TRUE"
# 
# $$\frac{True \ Positives}{True \ Positives + False \ Negatives}$$

# In[ ]:


def recall(metrices):
    return metrices['TP'] / (metrices['TP'] + metrices['FN'])

metrices['recall'] = recall(metrices)
metrices['recall']


# ## F1 Score
# F1 Score is the weighted average of the true postive rate(recall) and precision
# 
# $$2 * \frac{Precision \times Recall}{Precision + Recall}$$

# In[ ]:


def f1_score(metrices):
    return 2 * metrices['precision'] * metrices['recall'] / (metrices['precision'] + metrices['recall'])

metrices['f1_score'] = f1_score(metrices)
metrices['f1_score']


# ## True Positive Rate (TPR or Recall or Sensitivity)
# To scribe in words, TPR is "If it is TRUE, how often the model predicts TRUE"
# 
# $$\frac{True \ Positives}{True \ Positives + False \ Negatives}$$

# In[ ]:


def tpr(metrics):
    return metrics['TP'] / (metrics['TP'] + metrics['FN'])

metrices['TPR'] = tpr(metrices)
metrices['TPR']


# ## False Positive Rate (FPR)
# To scribe in words, FPR is "When it is FALSE, how often the model predicts TRUE"
# 
# $$\frac{False \ Positives}{False \ Positives + True \ Negatives}$$

# In[ ]:


def fpr(metrics):
    return metrics['FP'] / (metrics['FP'] + metrics['TN'])

metrices['FPR'] = fpr(metrices)
metrices['FPR']


# Let us run the test multiple times and observe how the metrices we have learnt comes out for digit recognition dataset.

# In[ ]:


def run(iterations=5):    
    outcomes = {'precision': [], 'recall': [], 'f1_score':[], 'tpr': [], 'fpr': []}
    
        
    for index in np.arange(1,iterations):
        print("Iternation {0}".format(index))
        classification = incomplete_model(X_train, y_train, X_test, index)
        sample_indices = np.random.randint(low=0, high=y_test.shape[0], size=index * 50)

        actual_sample = y_test[sample_indices]
        predicted_sample = classifications[sample_indices]

        metrices = confusion_matrix(predicted_sample, actual_sample)
        metrics = Metrics(metrices)
        metrics.calculate()
        outcomes['precision'].append(metrics.precision)
        outcomes['recall'].append(metrics.recall)
        outcomes['f1_score'].append(metrics.f1_score)
        outcomes['tpr'].append(metrics.tpr)
        outcomes['fpr'].append(metrics.fpr)
    return outcomes
    
outcomes = run(15)


# In[ ]:


df = pd.DataFrame(data=outcomes)
df


# In[ ]:


import matplotlib.pyplot as plt
from PIL import Image

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 9999
pd.options.display.float_format = '{:20, .2f}'.format


# In[ ]:


df = df.sort_values(by ='recall')

trace = go.Scatter(x = df.recall, y = df.precision)
data = [trace]
py.iplot(data, filename='AUC')


# ## PR Curve (Work In Progress)
# - Precision - Recall curves gives more informative picture of an algorithm's performance
# - Visibly it looks like convex hull of the ROC curve's space
# 
# $$Recall(FPR) \ vs \ Precision$$
# 
# - In PR Space, the goal is to e in the upper right hand corner

# ## ROC Curve (Work In Progress)
# ROC stands for **Receiver Operator Characteristics**, ROC curves are used to present results of binary decision problems. 
# - ROC curves shows the number of correctly classified positive samples varies with the number of incorrectly classified negative examples. ie 
# 
# $$False \ Positive \ Rate(FPR) \ vs \ True \ Positive \ Rate(TPR)$$
# 
# - ROC Curves can present an overly optimistic view of an algorithm's performance
# - In ROC space, the goal is to be in the upper left hand corner
# 

# In[ ]:


df = df.sort_values(by ='fpr')
 
trace = go.Scatter(x = df.fpr, y = df.tpr)
data = [trace]
py.iplot(data, filename='ROC')

