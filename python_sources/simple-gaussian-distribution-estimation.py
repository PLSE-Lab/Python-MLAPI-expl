#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection
# 
# In the following notebook we will attempt to perform anomaly detection on the labelled credit card dataset.
# The dataset consists of 284,807 observations and 31 features of which 1 is the class label.
# 
# 
# We will attempt to model our data using a multivariate Gaussian distribution and flag transactions
# as anomalous if the likelihod that they come from the modelled distribution is below a set threshold.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.stats as stats

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)
    for filename in filenames:
        credit_data = pd.read_csv(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


credit_data.shape


# In[ ]:


credit_data.describe()


# # Next, split data and remove non-used columns

# In[ ]:


from sklearn.model_selection import train_test_split

def split_data(credit_data):
    """
    Splits the data into sets of size 0.6, 0.2 and 0.2
    where the train set does not contain any outliers and the
    validation and test sets both contain the same amount
    of outliers.
    """
    
    # Separate the outliers because we want
    # both sets to have the same amount of outliers
    non_outliers = credit_data.loc[credit_data['Class'] == 0]
    outliers = credit_data.loc[credit_data['Class'] == 1]
    
    # Split the data
    train, test = train_test_split(non_outliers, test_size=0.4, random_state=42)
    val, test = train_test_split(test, test_size=0.5, random_state=42)
    
    
    val_outliers, test_outliers = train_test_split(outliers, test_size=0.5, random_state=42)
    
    # Add the outliers back to both sets
    val = pd.concat([val, val_outliers])
    test = pd.concat([test, test_outliers])
    
    return train, val, test

train, val, test = split_data(credit_data)

print(train['Class'].value_counts())
print(val['Class'].value_counts())
print(test['Class'].value_counts())


# In[ ]:


TRIVIAL_COLUMN_NAMES = ['Class', 'Time', 'Amount']
CLASS_COLUMN_NAME = 'Class'

def separate_x_y(data):
    X = data.drop(columns=TRIVIAL_COLUMN_NAMES)
    y = data[CLASS_COLUMN_NAME]
    
    return X,y


# In[ ]:


X_train, _ = separate_x_y(train)
X_val, y_val = separate_x_y(val)
X_test, y_test = separate_x_y(test)


# # For learning purposes, we will write our own classifier that works as described in the introduction

# In[ ]:


from sklearn.base import BaseEstimator, ClassifierMixin

class GaussianAnomalyClassifier(BaseEstimator, ClassifierMixin):
    """
    An anomaly detection classifier.
    """
    
    ANOMALY_CLASS_LABEL = 1
    NON_ANOMALY_CLASS_LABEL = 0
    
    def __init__(self, anomaly_threshold=None):
        """
        params:
        anomaly_threshold - The minimum probability a sample
                            can have before being classified
                            as an anomaly.
        """
        
        self.anomaly_threshold = anomaly_threshold
        
        
    def fit(self, X, y=None):
        """
        Estimates the parameters of a multivariate Gaussian
        distribution on X.
        """
        covariance_matrix = np.cov(X, rowvar=0)
        means = np.mean(X,axis=0)
        
        self.distribution = stats.multivariate_normal(mean=means, cov=covariance_matrix)
        
        
        return self
    
    def predict_proba(self, X):
        """
        Calculates the likelihoods of X
        coming from the estimated distribution
        """
        if self.distribution is None:
            raise RuntimeError("You must train the classifier before prediction")
            
        probabilities = self.distribution.pdf(X)
        
        return probabilities
    
    def predict(self, X, y=None):
        """
        Classifies each sample in X
        """
        
        probabilities = self.predict_proba(X)
        
        predictions = np.where(probabilities < self.anomaly_threshold,                                self.ANOMALY_CLASS_LABEL, self.NON_ANOMALY_CLASS_LABEL)
        
        return predictions
        


# # Below, we will try to automatically find a threshold using Scikit-Learns precision_recall_curve function

# In[ ]:


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

classifier = GaussianAnomalyClassifier(0.01)
classifier = classifier.fit(X_train)
predictions = classifier.predict_proba(X_val)

precision_recall = precision_recall_curve(y_val, predictions)

plt.figure()
plt.step(precision_recall[1],precision_recall[0], where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 0.1])
plt.xlim([0.0, 1.0])


# # The above automatic threshold tester precision_recall_curve from Scikit-Learn does not seem to give good results.
# 
# # So, we will manually test out different thresholds below.

# In[ ]:


from sklearn.metrics import precision_recall_fscore_support

def print_scores(predictions, y_true):
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, predictions, average='binary', pos_label=1)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('Fscore: ', fscore)
    print()
    print(pd.Series(predictions).value_counts())
    
    return precision,recall,fscore


# In[ ]:


print("REAL VALUE COUNTS VALIDATION SET")
print(pd.Series(y_val).value_counts())
print('------------------------------')
print('------------------------------\n')


# create a list of increasingly smaller thresholds to test
thresholds = [0.000000000000000001 * (0.1)**x for x in range(170)]
thresholds.reverse()

counter = 159
fscores = [] # Save fscores to plot them afterwards

for threshold in thresholds:
    classifier = GaussianAnomalyClassifier(threshold)
    classifier = classifier.fit(X_train)
    predictions = classifier.predict(X_val)

    _,_, fscore = print_scores(predictions, y_val)
    
    print('threshold index: ', counter)
    print('------------------------')
    
    
    fscores.append(fscore)
    counter -= 1
    
fscores.reverse()


# In[ ]:


plt.scatter(range(len(fscores)), fscores, s=4)

plt.title("Fscores of increasingly smaller probability thresholds")
plt.xlabel('Threshold Index')
plt.ylabel('Fscore')


# # Catching fraudulent transactions is more important than not wrongly flagging valid transactions. So, we will chose a threshold with a higher recall at the cost of some precision.
# 
# However, recall seems to plateau below index 80 where the recall is approximately 0.75.
# Therefore, we will choose the threshold with index 80.
# 
# The following statistics come from index 80:
# 
#     Precision:  0.5210084033613446
#     Recall:  0.7560975609756098
#     Fscore:  0.6169154228855721
# 
#     0    56752
#     1      357
#     dtype: int64
#     threshold index:  80
#     
# index 80 corresponds to the threshold of: 
# # 0.000000000000000001 * (0.1)**80
# 

# # At last, we compute the final test set score

# In[ ]:


threshold = 0.000000000000000001 * (0.1)**80

classifier = GaussianAnomalyClassifier(threshold)
classifier = classifier.fit(X_train)
predictions = classifier.predict(X_test)

_,_,_ = print_scores(predictions, y_test)


# # Our final test evaluation results in just a slightly lower Fscore.
# 
# # Our test score forecasts that we will catch about 79% of fraudulent transactions and that approximately half of the transaction we flag as fraudulent are actually normal, non fraudulent transactions 
