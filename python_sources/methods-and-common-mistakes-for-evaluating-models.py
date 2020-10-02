#!/usr/bin/env python
# coding: utf-8

# # cross-validation and resampling for heavily imbalanced data
# 
# In this notebook, I will demonstrate good methodology for evaluating a classifier on this data set using k-fold cross-validation.  I write this notebook because many of the top rated kaggle kernels for this data set have major overfitting problems or use the wrong metrics.
# 
# We will only evaluate a logistic regression model for the sake of example. The code is flexible enough  that you can use it to evaluate whatever other Scikit-learn classifier (or wrapped classifier) you like.  
# 
# We experiment with over/undersampling using the different samplers available from the imblearn package.
# 
# ### Methods and common mistakes
# 
# - **Area under the precision-recall curve:** In this notebook, we choose AUPRC as our evaluation metric, as recommended on the [kaggle dataset page](https://www.kaggle.com/mlg-ulb/creditcardfraud) for heavily imbalanced datasets with few positive samples where the goal is to detect as many of the postive examples as possible.  This is explained nicely i [this blog post](https://towardsdatascience.com/what-metrics-should-we-use-on-imbalanced-data-set-precision-recall-roc-e2e79252aeba). AUPRC is not one of the metrics directly included in sklearn, so we create our own function.
# 
# - **Do not over-sample before k-fold cross-validation:** Over-sampling before performing the k-fold split introduces identical samples in the training and test sets for each fold, which leads to an overestimate of model performance.  This is explained very nicely in this [blog post](https://www.marcoaltini.com/blog/dealing-with-imbalanced-data-undersampling-oversampling-and-proper-cross-validation).  There are various different methods of over/under sampling provided by the imblearn package. We compare these methods. Over/under-sampling is also crucial when training neural networks with sgd.
# 
# - **Do not do feature selection before cross-validation:** In discussions about the creditcard dataset several people point out that some models seem to perform better or worse when the Time feature is excluded. One may also observe from distribution plots that Time does not appear to be correlated with Fraud.  However, using a distribution plot or any other analysis of the whole data set to perform feature selection before performing the cross-validation split will result in overestimating model performance, so we do not do it here.
# 
# - **Don't test on resampled data:** Although we use resampling for the training sets, test sets should be as representative as the "true" distribution as possible.  
# 
# - **Use stratified k-folds:** Since there are so few positive examples in the dataset (~400/300,000), it is important that each fold has a representative number of positive examples. 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# functions needed for pr_auc_score()
from sklearn.metrics import auc, precision_recall_curve

# functions needed for imbalanced_cross_validation_score()
from sklearn.model_selection import StratifiedKFold

# sampler objects
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# Classification models to compare
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


# In[5]:


# load data
df = pd.read_csv("../input/creditcard.csv")
A  = df.values
x  = A[:,:-1]     
y  = A[:,-1]


# In[6]:


def pr_auc_score(clf, x, y):
    '''
        This function computes area under the precision-recall curve. 
    '''
      
    precisions, recalls,_ = precision_recall_curve(y, clf.predict_proba(x)[:,1], pos_label=1)
    
    return auc(recalls, precisions)


# In[14]:


def imbalanced_cross_validation_score(clf, x, y, cv, scoring, sampler):
    '''
        This function computes the cross-validation score of a given 
        classifier using a choice of sampling function to mitigate 
        the class imbalance, and stratified k-fold sampling.
        
        The first five arguments are the same as 
        sklearn.model_selection.cross_val_score.
        
        - clf.predict_proba(x) returns class label probabilities
        - clf.fit(x,y) trains the model
        
        - x = data
        
        - y = labels
        
        - cv = the number of folds in the cross validation
        
        - scoring(classifier, x, y) returns a float
        
        The last argument is a choice of random sampler: an object 
        similar to the sampler objects available from the python 
        package imbalanced-learn. In particular, this 
        object needs to have the method:
        
        sampler.fit_sample(x,y)
        
        See http://contrib.scikit-learn.org/imbalanced-learn/
        for more details and examples of other sampling objects 
        available.  
    
    '''
    
    cv_score = 0.
    train_score = 0.
    test_score = 0.
    
    # stratified k-fold creates folds with the same ratio of positive 
    # and negative samples as the entire dataset.
    
    skf = StratifiedKFold(n_splits=cv, random_state=0, shuffle=False)
    
    for train_idx, test_idx in skf.split(x,y):
        
        xfold_train_sampled, yfold_train_sampled = sampler.fit_sample(x[train_idx],y[train_idx])
        clf.fit(xfold_train_sampled, yfold_train_sampled)
        
        train_score = scoring(clf, xfold_train_sampled, yfold_train_sampled)
        test_score  = scoring(clf, x[test_idx], y[test_idx])
        
        print("Train AUPRC: %.2f Test AUPRC: %.2f"%(train_score,test_score))

        cv_score += test_score
        
    return cv_score/cv


# ## Basic models
# 
# Let's compare several basic models with different types of 
# over/under sampling. We will use 
# 
#     RandomOverSampler()
# 
#     SMOTE()
# 
#     ADASYN() 
# 
#     RandomUnderSampler() 
# 
# Documentation about these samplers can be found here:
# http://contrib.scikit-learn.org/imbalanced-learn/
# 
# We use 5-fold validation for all our tests, for the sake of 
# speed. One could also try 10-fold with a bit of patience.

# In[9]:


cv = 5  

RegressionModel    = LogisticRegression()

# here are some other models you could try. You can also try grid searching their hyperparameters
RandomForrestModel = RandomForestClassifier()
ExtraTreesModel    = ExtraTreesClassifier()
AdaBoostModel      = AdaBoostClassifier()


# In[ ]:


# Logistic regression score with Random Over-sampling
print("Random over-sampling")
score = imbalanced_cross_validation_score(RegressionModel, x, y, cv, pr_auc_score, RandomOverSampler())
print("Cross-validated AUPRC score: %.2f"%score)

# Logistic regression score with SMOTE
print("SMOTE over-sampling")
score = imbalanced_cross_validation_score(RegressionModel, x, y, cv, pr_auc_score, SMOTE())
print("Cross-validated AUPRC score: %.2f"%score)

# Logistic regression score with ADASYN
print("ADASYN over-sampling")
score = imbalanced_cross_validation_score(RegressionModel, x, y, cv, pr_auc_score, ADASYN())
print("Cross-validated AUPRC score: %.2f"%score)

# Logistic regression score with Random Under Sampling
print("Random under-sampling")
score = imbalanced_cross_validation_score(RegressionModel, x, y, cv, pr_auc_score, RandomUnderSampler())
print("Cross-validated AUPRC score: %.2f"%score)


# In[ ]:


# for fun, let's plot one of the precision-recall curves that is computed above
#
sampler = SMOTE()
skf = StratifiedKFold(n_splits=cv, random_state=0, shuffle=False)
clf = RegressionModel

train_idx, test_idx = skf.split(x,y).__next__()
xfold_train_sampled, yfold_train_sampled = sampler.fit_sample(x[train_idx],y[train_idx])

clf.fit(xfold_train_sampled, yfold_train_sampled)

precisions, recalls,_ = precision_recall_curve(y[test_idx], clf.predict_proba(x[test_idx])[:,1], pos_label=1)

plt.step(recalls, precisions, color='b', alpha=0.2,
         where='post')
plt.fill_between(recalls, precisions, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve')

