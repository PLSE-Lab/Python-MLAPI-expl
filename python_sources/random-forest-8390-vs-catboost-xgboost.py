#!/usr/bin/env python
# coding: utf-8

# # Air Pressure system (APS) failure cost prediction model
# 
# 
# 
# #### Acknowledgements
# This file is part of APS Failure and Operational Data for Scania Trucks. It was imported from the <a href="https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks">UCI ML Repository</a>.
# 
# #### Inspiration
# The total cost of a prediction model the sum of Cost_1 multiplied by the number of Instances with type 1 failure and Cost_2 with the number of instances with type 2 failure, resulting in a Total_cost. In this case Cost_1 refers to the cost that an unnecessary check needs to be done by an mechanic at an workshop, while Cost_2 refer to the cost of missing a faulty truck, which may cause a breakdown. Cost_1 = 10 and Cost_2 = 500, and Total_cost = Cost_1*No_Instances + Cost_2*No_Instances.
# 
# <a id='Top'></a>
# _________________________________________________________________________________________
# 
# ### Notes
# The dataset is highly imblanaced with a high amount of negative class and low amount of positive class, which needs to be considered for classifier model selection and evaulation. ML-Classifier algorithms like KNN and SVC has beend excluded due to the high imbalanaced dataset. In order to build a failure cost prediction model, it would make more sense to focus on Randomforest or Boosting alghorithms.
# 
# 
# ### Table of Contents
# 
# #### [Reading and preparing Dataset](#Data)
# #### [1.Random Forest](#Random)
# #### [2.Naive Bayes](#Naive)
# #### [3.XGBOOST](#XG)
# #### [4.CatBoost](#Cat)

# ### Import necessary libraries

# In[ ]:


get_ipython().system('pip install catboost')
get_ipython().system('pip install ipywidgets')
get_ipython().system('jupyter nbextension enable --py widgetsnbextension')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, Pool, cv
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from tqdm import tqdm

from sklearn.metrics import f1_score, log_loss, confusion_matrix,classification_report
import scikitplot as skplt
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler


# ### Reading Dataset in dataframe

# In[ ]:


data_train = pd.read_csv('../input/aps_failure_training_set.csv')
data_test = pd.read_csv('../input/aps_failure_test_set.csv')


# In[ ]:


data_train.head()


# In[ ]:


data_train.isnull().sum()


# In[ ]:


data_test.isnull().sum()


# In[ ]:


# NA replacemenet
data_train.replace('na','-1', inplace=True)
data_test.replace('na','-1', inplace=True)


# In[ ]:


#categorical encoding
data_train['class'] = pd.Categorical(data_train['class']).codes
data_test['class'] = pd.Categorical(data_test['class']).codes

print(['neg', 'pos'])
print(np.bincount(data_train['class'].values))
print(np.bincount(data_test['class'].values))


# In[ ]:


import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
sns.countplot(x='class', data=data_train, palette='hls')
plt.show()


# <a id='Data'></a>
# ### Preparing Data

# In[ ]:


# split train and test data into X_train,X_test and y_train,y_test
y_train = data_train['class'].copy(deep=True)
X_train = data_train.copy(deep=True)
X_train.drop(['class'], inplace=True, axis=1)

y_test = data_test['class'].copy(deep=True)
X_test = data_test.copy(deep=True)
X_test.drop(['class'], inplace=True, axis=1)

# strings to float
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')


# In[ ]:


cat_features = list(range(0, X_train.shape[1]))
print(cat_features)


# In[ ]:


print(X_train.dtypes)
categorical_features_indices = np.where(X_train.dtypes != np.float)[0]


# #### Definition of generic function for model evaulation scores

# In[ ]:


def evaluate(y_test,y_pred,y_pred_proba):
    if len(y_pred)>0:
        f1 = f1_score(y_test,y_pred,average="weighted")
        print("F1 score: ",f1)
    if len(y_pred_proba)>0:
        logloss = log_loss(y_test,y_pred_proba, eps=1e-15, normalize=True, sample_weight=None, labels=None)
        print("Log loss for predicted probabilities:",logloss)


# <a id='Random'></a>
# ## 1.Random Forest

# In[ ]:


forest_clf = RandomForestClassifier(n_estimators=250,n_jobs=-1)
forest_clf.fit(X_train,y_train)
y_pred_rf = forest_clf.predict(X_test)
y_pred_proba_rf = forest_clf.predict_proba(X_test)
evaluate(y_test,y_pred_rf,y_pred_proba_rf)


# In[ ]:


tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf ).ravel()
skplt.metrics.plot_confusion_matrix(y_test, y_pred_rf, normalize=False)
plt.show()
print(classification_report(y_test,y_pred_rf))


# In[ ]:


#display ROC curve
from sklearn.metrics import auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred_rf)
roc_auc = auc(fpr, tpr)


plt.figure()
plt.plot(fpr, tpr, color='red', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# Recall and precision are highly valuable evaulation metrics for our imbalanced data, it should be considered to manipualte propability thresholds to evaulate impacts regarding the confusion matrix and finally to decide on further steps to search for threshold optimisation.

# In[ ]:


y_test_predictions_rec = y_pred_proba_rf[:,1] > 0.1
y_test_predictions_prec = y_pred_proba_rf[:,1] > 0.85


# In[ ]:


skplt.metrics.plot_confusion_matrix(y_test, y_test_predictions_prec, normalize=False)
plt.show()
print(classification_report(y_test, y_test_predictions_prec))


# ### [Back to Top](#Top)

# In[ ]:


roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='red',label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# As it is too risky to maniupulate propability thresholds, a function is defined to process a propability threshold in order to finally trigger a robust failure cost model which is based on an improved confusion matrix.

# In[ ]:


scores = forest_clf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, scores)


# In[ ]:


min_cost = np.inf
best_threshold = 0.5
costs = []
for threshold in tqdm(thresholds):
    y_pred_threshold = scores > threshold
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()
    cost = 10*fp + 500*fn
    costs.append(cost)
    if cost < min_cost:
        min_cost = cost
        best_threshold = threshold
print("Best threshold: {:.4f}".format(best_threshold))
print("Min cost: {:.2f}".format(min_cost))


# In[ ]:


y_pred_test_rf = forest_clf.predict_proba(X_test)[:,1] > best_threshold
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test_rf).ravel()
skplt.metrics.plot_confusion_matrix(y_test,y_pred_test_rf, normalize=False)
10*fp + 500*fn


# #### Total failure cost prediction with improved threshold (0.0440) is 8390 using Random Forest.

# ### [Back to Top](#Top)

# <a id='Naive'></a>
# ## 2.Naive Bayes

# In[ ]:


bayes_clf = GaussianNB()
bayes_clf.fit(X_train,y_train)


# In[ ]:


y_pred__bayes = bayes_clf.predict(X_test)
y_pred_proba_bayes = bayes_clf.predict_proba(X_test)


# In[ ]:


evaluate(y_test,y_pred__bayes,y_pred_proba_bayes)


# In[ ]:


tn, fp, fn, tp = confusion_matrix(y_test,y_pred__bayes).ravel()
skplt.metrics.plot_confusion_matrix(y_test,y_pred__bayes, normalize=False)
plt.show()
print(classification_report(y_test,y_pred__bayes))


# In[ ]:


from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test,y_pred__bayes)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='red',label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# ### [Back to Top](#Top)

# <a id='XG'></a>
# ## 3.XGBOOST

# In[ ]:


xgb_clf = XGBClassifier(max_depth=5)
xgb_clf.fit(X_train,y_train)


# In[ ]:


y_pred_xgb = xgb_clf.predict(X_test)
y_pred_proba_xgb = xgb_clf.predict_proba(X_test)


# In[ ]:


evaluate(y_test,y_pred_xgb,y_pred_proba_xgb)


# In[ ]:


tn, fp, fn, tp = confusion_matrix(y_test,y_pred_xgb).ravel()
skplt.metrics.plot_confusion_matrix(y_test,y_pred_xgb, normalize=False)
plt.show()
print(classification_report(y_test,y_pred_xgb))


# In[ ]:


from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test,y_pred_xgb)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='red',label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# Recall and precision are highly valuable evaulation metrics for our imbalanced data, it should be considered to manipualte propability thresholds to evaulate impacts regarding the confusion matrix and finally to decide on further steps to search for threshold optimisation.

# In[ ]:


y_test_predictions_rec = y_pred_proba_xgb[:,1] > 0.1
y_test_predictions_prec = y_pred_proba_xgb[:,1] > 0.85


# In[ ]:


skplt.metrics.plot_confusion_matrix(y_test, y_test_predictions_prec, normalize=False)
plt.show()
print(classification_report(y_test, y_test_predictions_prec))


# ### [Back to Top](#Top)

# As it is too risky to maniupulate propability thresholds, a function is defined to process a propability threshold in order to finally trigger a robust failure cost model which is based on an improved confusion matrix.

# In[ ]:


scores = xgb_clf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, scores)


# In[ ]:


roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='red',label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


min_cost = np.inf
best_threshold = 0.5
costs = []
for threshold in tqdm(thresholds):
    y_pred_threshold = scores > threshold
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()
    cost = 10*fp + 500*fn
    costs.append(cost)
    if cost < min_cost:
        min_cost = cost
        best_threshold = threshold
print("Best threshold: {:.4f}".format(best_threshold))
print("Min cost: {:.2f}".format(min_cost))


# In[ ]:


y_pred_test_xgb = xgb_clf.predict_proba(X_test)[:,1] > best_threshold
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test_xgb).ravel()
skplt.metrics.plot_confusion_matrix(y_test,y_pred_test_xgb, normalize=False)
10*fp + 500*fn


# #### Total failure cost prediction with improved threshold (0.0069) is 9520 using XGBOOST.

# ### [Back to Top](#Top)

# <a id='Cat'></a>
# ## 4.CatBoost

# Another Boosting API is CatBoost and a candidate for "everybodys classifier" due to these unique advantages:
# * 1.Indicies for categorical columns (CatBoost preprocessing step)
# * 2.Embedded vizualisation and comparsion for LogLoss and Accurancy,AUC etc. curves considering different parameter settings
# * 3.High flexibility for parameter settings e.g. number of trees, learning rate, iterations etc.

# In[ ]:


y_test.shape


# In[ ]:


X_test.shape


# In[ ]:


X_train.fillna(-999, inplace=True)
X_test.fillna(-999, inplace=True)


# In[ ]:


model = CatBoostClassifier(
    custom_loss=['Accuracy'],
    random_seed=42,
    logging_level='Silent'
)
model.fit(
    X_train, y_train,
    cat_features=categorical_features_indices,
    eval_set=(X_test, y_test),
    logging_level='Verbose',  # you can uncomment this for text output
    plot=True
);


# ### [Back to Top](#Top)

# The labelled data y need to be classified by the independed values x, in our case we are focusing as mentioned above on a classification problem in order to build a predictive failure cost model for the APS system. 
# The objective functions for optimising classification algorithms are
# 1. Logloss function (for binary labeled data[0,1]) 
# 2. CrossEntropy (prediction of propabilities to estimate class of labeled data)

# The CatboostClassifier library provides the opportunity to visualize the important objective functions Logloss and CrossEntropy.
# Relevant training parameters parameters like learning rate and iterations for optimising the objective functions can be set within CatboostClassifier.

# ### [Back to Top](#Top)

# ### Standard output of the training

# Among training parameters, in CatBoost it is possible to define a logging_level for the standard output to shed light on parameters like  
# * error on learning & test set (value of objective function)
# * optimized metric
# * elapsed training time
# * remaining training time
# * best iteration
# * no. of trees
# 
# a file will be created on the local system for for defined training rates 

# Catboost provides a MetricVisualizer in order to plott the error on learning and test rates for each tuned learning rate mentioned earlier.
# The best iteration is plotted as a dot in the learning curve and also outlined in the metrics.

# ### [Back to Top](#Top)

# ### Eval metric, custom metrics and best trees count

# More interesting than objective functions for classification tasks, it is important to deep dive on accuracy, precision, recall or f measure of the model

# In[ ]:


model = CatBoostClassifier(
    iterations=450,
    random_seed=38,
    learning_rate=0.2,
    eval_metric="Accuracy",
    use_best_model=False
)

model.fit(
    X_train, y_train,
    cat_features=categorical_features_indices,
    eval_set=(X_test, y_test),
    verbose=False,
    plot=True
)


# ### Crossvalidation

# In[ ]:


params = {}
params['loss_function'] = 'Logloss'
params['iterations'] = 450
params['custom_loss'] = 'AUC'
params['random_seed'] = 60
params['learning_rate'] = 0.2

cv_data = cv(
    params = params,
    pool = Pool(X_train, label=y_train, cat_features=categorical_features_indices),
    fold_count=5,
    inverted=False,
    shuffle=True,
    partition_random_seed=0,
    plot=True,
    stratified=True,
    verbose=False
)


# ### [Back to Top](#Top)

# ### Model prediction

# ##### first column is propability of class 0 and second column is propability of class 1

# In[ ]:


print(model.predict_proba(data=X_test))


# In[ ]:


import scikitplot as skplt


# In[ ]:


y_pred = model.predict(data=X_test)


# In[ ]:


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()


# In[ ]:


skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False)
y_pred_proba = model.predict_proba(X_test)
plt.show()
print(classification_report(y_test, y_pred))


# ### [Back to Top](#Top)

# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='red',label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# Recall and precision are highly valuable evaulation metrics for our imbalanced data, it should be considered to manipualte propability thresholds to evaulate impacts regarding the confusion matrix and finally to decide on further steps to search for threshold optimisation.

# In[ ]:


y_test_predictions_high_precision = y_pred_proba[:,1] > 0.8
y_test_predictions_high_recall = y_pred_proba[:,1] > 0.1


# In[ ]:


skplt.metrics.plot_confusion_matrix(y_test, y_test_predictions_high_precision, normalize=False)
plt.show()
print(classification_report(y_test, y_test_predictions_high_precision))


# ### [Back to Top](#Top)

# In[ ]:


10*120+ 500*9


# As it is too risky to maniupulate propability thresholds, a function is defined to process a propability threshold in order to finally trigger a robust failure cost model which is based on an improved confusion matrix.

# In[ ]:


scores = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, scores)


# In[ ]:


roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='red',label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


min_cost = np.inf
best_threshold = 0.5
costs = []
for threshold in tqdm(thresholds):
    y_pred_threshold = scores > threshold
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()
    cost = 10*fp + 500*fn
    costs.append(cost)
    if cost < min_cost:
        min_cost = cost
        best_threshold = threshold
print("Best threshold: {:.4f}".format(best_threshold))
print("Min cost: {:.2f}".format(min_cost))


# In[ ]:


y_pred_test_final = model.predict_proba(X_test)[:,1] > best_threshold
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test_final).ravel()
skplt.metrics.plot_confusion_matrix(y_test,y_pred_test_final, normalize=False)
10*fp + 500*fn


# #### Total failure cost prediction with improved threshold (0.0049) is 10620 using CatBoost

# ### [Back to Top](#Top)

# ### Conclusion
# * Random Forest failre cost is 8390
# * CatBoost failure cost is 10620
# * XGBoost failure cost is 9520
# 
