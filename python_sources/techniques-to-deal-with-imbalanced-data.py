#!/usr/bin/env python
# coding: utf-8

# # Techniques to deal with imbalanced data
# 
# Imbalance in data is real world datasets is more a norm than an exception. Blinding application and optimization of learning algorithms on imbalance datasets can and will lead to semanctically incorrect models. Further traditional metric of model performance evaluation - accuracy often provides false indication of model correctness. In this kernel, we will highlight main techniques to dealing with imbalance. In the context of credit fraud dataset, we will explore efficacy of these tehcniques. 
# 
# NOTE:  Thanks to joparga3 (https://www.kaggle.com/joparga3/in-depth-skewed-data-classif-93-recall-acc-now) for the inspiration. Thanks to scikit for providing all the tools needed to tackle imbalance (http://contrib.scikit-learn.org/imbalanced-learn/stable/user_guide.html)
# 
# NOTE: Kernel runtime is quite high (comment out hyperparam tuning to speedup)
# 
# ## Pre-processing techniques
# #### Sampling techniques
# These techniques are part of pre-processing stage of ML pipelines prior to feeding data to training stage.
# 1. Undersampling
#     * Random undersampling
#     * ClusterCentroids
#     * NearMiss
# 2. Oversampling
#     * Random oversampling: generates new samples by random resampling with replacement of under represented class
#     * Synthetic Minority Oversampling (SMOTE)
# 3. Combined over and under sampling
#     * SMOTEENN
#     * SMOTETomek
# 
# ## Training techniques
# Number of learning models themselves do provide some built in support to deal with imbalance data.   
# 1.  Class weighting
# 2. Sample weighting

# ## Setup imports

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.metrics import recall_score,accuracy_score,confusion_matrix, f1_score, precision_score, auc,roc_auc_score,roc_curve, precision_recall_curve
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import ClusterCentroids,NearMiss, RandomUnderSampler
from imblearn.combine import SMOTEENN,SMOTETomek
from imblearn.ensemble import BalanceCascade
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ## Load the dataset

# In[ ]:


data = pd.read_csv("../input/creditcard.csv")
data.head()


# ## Check magnitude of imbalance

# In[ ]:


count_classes = pd.value_counts(data['Class'], sort = True).sort_index()
print(count_classes)


# In[ ]:


data['Amount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))
data = data.drop(['Time'],axis=1)
data.head()


# ## Split into training and test datasets

# In[ ]:


X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
print(X_train.shape)
print(X_test.shape)


# ## Helper functions

# In[ ]:


def benchmark(sampling_type,X,y):
    lr = LogisticRegression(penalty = 'l1')
    param_grid = {'C':[0.01,0.1,1,10]}
    gs = GridSearchCV(estimator=lr, param_grid=param_grid, scoring='accuracy', cv=5, verbose=2)
    gs = gs.fit(X.values,y.values.ravel())
    return sampling_type,gs.best_score_,gs.best_params_['C']

def transform(transformer,X,y):
    print("Transforming {}".format(transformer.__class__.__name__))
    X_resampled,y_resampled = transformer.fit_sample(X.values,y.values.ravel())
    return transformer.__class__.__name__,pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)


# ## Apply transformations to dataset

# In[ ]:


datasets = []
datasets.append(("base",X_train,y_train))
datasets.append(transform(SMOTE(n_jobs=-1),X_train,y_train))
datasets.append(transform(RandomOverSampler(),X_train,y_train))
#datasets.append(transform(ClusterCentroids(n_jobs=-1),X_train,y_train))
datasets.append(transform(NearMiss(n_jobs=-1),X_train,y_train))
datasets.append(transform(RandomUnderSampler(),X_train,y_train))
datasets.append(transform(SMOTEENN(),X_train,y_train))
datasets.append(transform(SMOTETomek(),X_train,y_train))




# ## Determine best hyperparameters

# In[ ]:


benchmark_scores = []
for sample_type,X,y in datasets:
    print('______________________________________________________________')
    print('{}'.format(sample_type))
    benchmark_scores.append(benchmark(sample_type,X,y))
    print('______________________________________________________________')
    


# In[ ]:


benchmark_scores


# ## Train/evaluate models for each of tranformed datasets

# In[ ]:


scores = []
# train models based on benchmark params
for sampling_type,score,param in benchmark_scores:
    print("Training on {}".format(sampling_type))
    lr = LogisticRegression(penalty = 'l1',C=param)
    for s_type,X,y in datasets:
        if s_type == sampling_type:
            lr.fit(X.values,y.values.ravel())
            pred_test = lr.predict(X_test.values)
            pred_test_probs = lr.predict_proba(X_test.values)
            probs = lr.decision_function(X_test.values)
            fpr, tpr, thresholds = roc_curve(y_test.values.ravel(),pred_test)
            p,r,t = precision_recall_curve(y_test.values.ravel(),probs)
            scores.append((sampling_type,
                           f1_score(y_test.values.ravel(),pred_test),
                           precision_score(y_test.values.ravel(),pred_test),
                           recall_score(y_test.values.ravel(),pred_test),
                           accuracy_score(y_test.values.ravel(),pred_test),
                           auc(fpr, tpr),
                           auc(p,r,reorder=True),
                           confusion_matrix(y_test.values.ravel(),pred_test)))


# ## Tabulate results

# In[ ]:


sampling_results = pd.DataFrame(scores,columns=['Sampling Type','f1','precision','recall','accuracy','auc_roc','auc_pr','confusion_matrix'])
sampling_results


# ## Train model with weighted class

# In[ ]:


lr = LogisticRegression(penalty = 'l1',class_weight="balanced")
lr.fit(X_train.values,y_train.values.ravel())
scores = []
pred_test = lr.predict(X_test.values)
pred_test_probs = lr.predict_proba(X_test.values)
probs = lr.decision_function(X_test.values)
fpr, tpr, thresholds = roc_curve(y_test.values.ravel(),pred_test)
p,r,t = precision_recall_curve(y_test.values.ravel(),probs)
scores.append(("weighted_base",
                           f1_score(y_test.values.ravel(),pred_test),
                           precision_score(y_test.values.ravel(),pred_test),
                           recall_score(y_test.values.ravel(),pred_test),
                           accuracy_score(y_test.values.ravel(),pred_test),
                           auc(fpr, tpr),
                           auc(p,r,reorder=True),
                           confusion_matrix(y_test.values.ravel(),pred_test)))

scores = pd.DataFrame(scores,columns=['Sampling Type','f1','precision','recall','accuracy','auc_roc','auc_pr','confusion_matrix'])


# ## Compare alternatives

# In[ ]:


results = sampling_results.append(scores)
results


# ## Interpretation
# *  Undersampling leads to high recall but comes at a huge cost to precision (also reducing training time)
# *  SMOTE(basic, Tomek,ENN) sampling and RandomOverSampler perform the best considering auc_roc and auc_pr with accpetables levels of false positives
# *  Class weighting seems to produce comparable results to sampling techniques (for models that support such weighting)

# ## Todos
# * Add best practicies
# 
# ## Extensions
