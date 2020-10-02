#!/usr/bin/env python
# coding: utf-8

# **Welcome**
# Thank you for reading, if you like please upvote!.
# 
# **Introduction**  
# In this Kernel, we study several methods to sample the data, and how this affects the time and the accuracy of our model.
# 
# The dataset has Credit Card transactions and our aim is to detect which are fraudulent.  
# 
# This is a binary classification problem (Fraud or Legal) 
# 
# The model selected is **Random Forest Classifier**.
# 
# 

# In[ ]:


#General Imports
import numpy as np # linear algebra
import pandas as pd # data processing


# In[ ]:


data = pd.read_csv('../input/creditcard.csv')
data.describe()


# Let's take a first look in our data.

# In[ ]:


data.columns


# In[ ]:


data.shape


# We have 284.807 records, with 31 columns. The columns V1 to V28 were processed with PCA. 
# 
# PCA (Principal Component Analysis) is a statistical method that reduces the number of columns (dimension) losing the minimal meaning of the original matrix.
# 
# 
# As wikipedia said "... its operation can be thought of as revealing the internal structure of the data in a way that best explains the variance in the data... PCA can supply the user with a lower-dimensional picture, a projection of this object when viewed from its most informative viewpoint..."

# In[ ]:


data.isnull().any().sum()


# Great, there is no Null values!

# **Imbalanced Data**
# 

# In[ ]:


LEGAL, FRAUD = range(2)

# sampler name constants
IMBALANCE, UNDER_RANDOM, OVER_RANDOM, OVER_SMOTE =  (
    'Imbalance', 'Random Under Sampler', 
    'Random Over Sampler','Smote'
)
n = data.Class.count()

frauds = data.Class == FRAUD
legals = data.Class == LEGAL
n_frauds = frauds.sum()
n_legals = legals.sum()

print('Total transactions:', n)
print('Legal transactions: {1} ({0:.4f}%).'
      ''.format(n_legals/n*100, n_legals))
print('Fraudulent transactions: {1} ({0:.4f}%).'
      ''.format(n_frauds/n*100, n_frauds))


# We need to train our model with a 50-50 balance data. However, there is a clear imbalance on the data. Due to the fact, most transactions are legal, as in real life. 
# 
# This could cause our model to make erroneous assumptions if we don't handle this imbalance.
# 
# In this kernel, we utilize several techniques to balance the training data and also will train 2 models with imbalanced data and compare the effects.

# In[ ]:


from collections import Counter

X = data.drop('Class', axis=1)
y = data.Class
c = Counter(y)
print('Original distribution '
      'Legal ({0}) - Fraud ({1}))'.format(c[LEGAL], c[FRAUD]))


# **Separate Train Test **

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
n = y_test.count()
n_frauds = y_test.sum()
"Fraud transactions: {0} ({1:.2}%)".format(n_frauds, n_frauds/n * 100)


# **Sampling**
# 

# In[ ]:


from time import time
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

samplers = {
    UNDER_RANDOM: RandomUnderSampler,
    OVER_RANDOM: RandomOverSampler,
    OVER_SMOTE: SMOTE,
}

samples = {IMBALANCE: (X_train, y_train)}
durations = {IMBALANCE: 0}

for name, sampler in samplers.items():
    start2 = time()
    smp = sampler(random_state=0)
    samples[name] = X_sample, y_sample = smp.fit_resample(X_train, y_train)
    durations[name] = time() - start2

    print('{0} tooks {1:.2} seconds'.format(name, durations[name]))
    print('Distribution is', Counter(y_sample))

    


# Now will train several RandomForest models with differents samples:
# * Random Under Sample 
# * Random Over Sample
# * Smote OverSample
# * Imbalanced Sample RF handling imbalance
# * Imbalanced Sample with No handling imbalance
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
models = dict()

common_args = {'n_estimators': 100, 
               'max_depth': 3,
               'random_state': 0, 'oob_score': True}
args = {
    IMBALANCE: {
        'class_weight': 'balanced_subsample' 
         }
}
models = dict()
samples[IMBALANCE+' No Handling'] = samples[IMBALANCE]
    
for name, sample in samples.items():
    specific_args = args.get(name, {})
    start = time()
    rfc = RandomForestClassifier(**common_args, **specific_args)
    rfc.fit(*sample)
    
    total_time = time() - start
    
    X_sample, y_sample = sample
    y_pred = rfc.predict(X_test)
    
    score = balanced_accuracy_score(y_test, y_pred)
    # store models and results
    models[name] = rfc, y_pred, score, total_time
    print('{0}\nSample Size: {3:.0f}\n'
          'Trained time: {1:.2f} seconds\n'
          'Balance Score: {2:.2f}%\n'
          'Oob Score: {4:.2f}%\n'.format(
            name, total_time, score*100, 
            len(y_sample), rfc.oob_score_*100))


# In[ ]:


MODEL, PREDICTS, SCORE, DURATION = range(4)
print('Out of Bag Scores')
for name in models.keys():
    score = models[name][MODEL].oob_score_
    
    print(' - {0} is {1:.2f}%'.format(name, score*100))


# **Confusion Matrix**
# 
# Following my intuition, I will use meaningful names instead of the general data science style ones. 
#  
# 
# - **True Positive**: Legal pass. Legal transaction that were correctly classified. These the ammout of client transaction working fine. 
# - **True Negative**: Fraud Blocked. Fraud detected and blocked.
# - **False Positive**: Legal Blocked. Legal transaction incorrectly classified (blocked).
# - **False Negative**: Fraud Pass. Fraud transaction incorrectly classified. No detected.
# 

# In[ ]:



from sklearn.metrics import confusion_matrix, classification_report
"""  
           LEGAL                    FRAUD
   PASS     lp(True Negative)       fp(False Negative)       
   BLOCK    lb(False Positive)      fb(True Positive)
   
    recall = fb / (fb+fp)        true pos / (true pos + false neg)
    precision = fb / (fb+lb)     true pos / (true pos + false pos)
    
    """
target_names = ['Legals', 'Frauds']
for name, model in models.items():
    # tn, fp, fn, tp
    # legal_passed, legal_blocked, fraud_passed, fraud_blocked 
    y_pred = model[PREDICTS]
    lp, lb, fp, fb = confusion_matrix(y_test, 
                                      y_pred).ravel()
    
    print('{0}:\nLegal Passed: {1}\n'
    'Legal Blocked: {2}\n'
    'Fraud Blocked: {4}\n'
    'Fraud Passed: {3}'
    ''.format(name, lp, lb, fp, fb))
    
    print('Detect {0:.2f}% Frauds and Block {1:.2f}% of legals\n'
          ''.format(fb/(fp+fb)*100, lb/(lb+lp)*100))
    
    print(classification_report(y_test, y_pred, target_names=target_names))


# **Conclusions**
# - **Random Under Sampler** has the best results detecting 87.19% of the frauds however has blocked 1.47% of legal transactions.
# This model was trained in 0.25seg, the fastest (sample 0.1s + train 0.15s).
# 
# - **Random Forest** has handled the imbalance of the data pretty well, detecting 85.71% of the frauds and blocking only 0.52% of legal transactions.  
# This model was trained in 40.69 seconds (162x times Random Under Sampler).
# 
# - You can see the consequences of training the model with imbalanced data (see **Imbalance No Handling** results) Only 60% of frauds transactions were detected, vs all other models with balanced data train had higher recall (approx 86% frauds detected).
# 

# **Next steps in futher versions**
# - Add Visualizations
# - Tuning Hyper parameters 
# - Try others Machine Learning: Support Vector Machine, Logistic Regression, Deep Learning.
# 
# Please let me your questions or comments, if you like my work, please upvote, Thank you.

# 
# 
