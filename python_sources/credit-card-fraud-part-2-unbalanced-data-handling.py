#!/usr/bin/env python
# coding: utf-8

# This is Part 2 - 
# In this part I'll show how the Near Miss and SMOTE CV improves the Recall Accuracy of the models as compared to the Stratified Kfold.
# 
# Please note that few steps like reading the file and normalizing/standardizing the Amount and Time columns will be repeated.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Importing all the necessary lib and the best estimated logestic Regression Model that was derived in Part 1.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score,roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold

log_r = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l1',
                   random_state=None, solver='warn', tol=0.0001, verbose=0,
                   warm_start=False)

df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
print(df.shape)
df.head()


# Using Robust Scaler over Standard scaler to Normalize the Amount and Time column, deleted the orignal columns in next step.

# In[ ]:



rscale = RobustScaler()
amount = df['Amount'].values.reshape(-1,1)
time = df['Time'].values.reshape(-1, 1)
df['scaled_amount'] = rscale.fit_transform(amount)
df['scaled_time'] = rscale.fit_transform(time)

df = df.drop(['Amount', 'Time'], axis = 1)

undersample_X = df.drop(['Class'], axis = 1)
undersample_y = df['Class']

undersample_X.shape


# IN the code below i used Stratified K fold validation method to find the accuracy and the Recall score. 

# In[ ]:


accuracy_skf = []
precision_score_skf = []
recall_score_skf = []
roc_auc_score_skf = []
f1_score_skf = []

skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
for train_index, test_index in skf.split(undersample_X, undersample_y):
#     print("TRAIN:", train_index, "TEST:", test_index)
    undersample_X_train, undersample_X_test = undersample_X.iloc[train_index], undersample_X.iloc[test_index]
    undersample_y_train, undersample_y_test = undersample_y.iloc[train_index], undersample_y.iloc[test_index]
    
    undersample_y_train = undersample_y_train.values.reshape(-1,1)
    undersample_y_test = undersample_y_test.values.reshape(-1, 1)
#     print('X - Train Shape', undersample_X_train.shape)
    log_r.fit(undersample_X_train, undersample_y_train)
    
#     print('X - Test Shape',undersample_X_test.shape)
#     prediction = log_r.predict(undersample_X_test)
    
    score_skf = accuracy_score(undersample_y_test, log_r.predict(undersample_X_test))
    accuracy_skf.append(score_skf)
    print('Accuracy SKF',score_skf)
    
    precision_skf = precision_score(undersample_y_test, log_r.predict(undersample_X_test))
    print('Precision Score SKF', precision_skf)
    precision_score_skf.append(precision_skf)
    
    recall_skf = recall_score(undersample_y_test, log_r.predict(undersample_X_test))
    print('Recall Score - SKF', recall_skf)
    recall_score_skf.append(recall_skf)
    
    f1_skf = f1_score(undersample_y_test, log_r.predict(undersample_X_test))
    print('F1 Score SKF',f1_skf)
    f1_score_skf.append(f1_skf)
    roc_auc_skf = roc_auc_score(undersample_y_test, log_r.predict(undersample_X_test))
    print('ROC Score SKF',roc_auc_skf)
    roc_auc_score_skf.append(roc_auc_skf)
    print(confusion_matrix(undersample_y_test, log_r.predict(undersample_X_test)))
    


# I just averaged out the scores from the 5 runs above. Interesting numbers are Accuracy, who would not want a model that is 99.9 % accurate. But is the model really good?
# Look the the Recall Score its too low, which means the fraud records that were ideally supposed to be predicted as Fraud are not done in a accurate way. 
# Sooo.. K stratified Cross Validation does not work for me.

# In[ ]:


import numpy as np
print('Accuracy - SKF',np.mean(accuracy_skf))
print('Precision Score - SKF',np.mean(precision_score_skf))
print('Recall Score - SKF',np.mean(recall_score_skf))
print('ROC-AUC - SKF',np.mean(roc_auc_score_skf))
print('F1 Score - SKF',np.mean(f1_score_skf))


# I will be trying the Near Miss and the SMOTE Cross Validation functions
# Near Miss - in my terms, this function will reduce the number of majority class to the number of minority class, it uses vector distance method to do so. It picks up records that are closest to the minority class data.
# SMOTE - This method adds minority class records. 

# In[ ]:


from imblearn.under_sampling import NearMiss

nm = NearMiss()
X_nearmiss,y_nearmiss = NearMiss().fit_sample(undersample_X, undersample_y)

from sklearn.model_selection import train_test_split
X_train_nm, X_test_nm, y_train_nm, y_test_nm = train_test_split(X_nearmiss, y_nearmiss)

log_r.fit(X_train_nm,y_train_nm)
prediction = log_r.predict(X_test_nm)
score_nm = accuracy_score(y_test_nm, prediction)
print('Accuracy NearMiss',score_nm)
precision_score_nm = precision_score(y_test_nm, prediction)
print('Precision Score NearMiss', precision_score_nm)
recall_score_nm = recall_score(y_test_nm, prediction)
print('Recall Score - NearMiss', recall_score_nm)
f1_score_nm = f1_score(y_test_nm, prediction)
print('F1 Score NearMiss',f1_score_nm)
roc_auc_score_nm = roc_auc_score(y_test_nm, prediction)
print('ROC Score NearMiss',roc_auc_score_nm)


# Yes, the accuracy has gone down a bit, but look at the Recall score go UP!! This model is better that the previous one for sure!
# Lets check if the SMOTE provides any better results.

# In[ ]:


from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
style.use('fivethirtyeight')
def plot_learning_curve(model, X, y):
    train_size, train_scores, test_scores = learning_curve(model, X, y, train_sizes=np.linspace(0.01, 1, 50), cv=10,
                                                       scoring='accuracy', n_jobs=3, verbose=1, random_state=42,
                                                      shuffle=True)
    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std = np.std(train_scores, axis = 1)
    test_scores_mean = np.mean(test_scores, axis = 1)
    test_scores_std = np.std(test_scores, axis = 1)
    plt.figure(figsize = (8, 4))
    plt.plot(train_size, train_scores_mean, color = 'red', label = 'Training Score')
    plt.fill_between(train_size, train_scores_mean - train_scores_std, train_scores_mean+train_scores_std, color = '#DDDDDD')
    plt.fill_between(train_size, test_scores_mean - test_scores_std, test_scores_mean+test_scores_std, color = '#DDDDDD')
    plt.plot(train_size, test_scores_mean, color = 'green', label = 'CV Score')
    plt.title('Learning Curve ')
    plt.xlabel('CV Train Size')
    plt.ylabel('Accuracy')
    plt.legend(loc = 'best')
    plt.show()
    
   


# In[ ]:


plot_learning_curve(log_r, X_nearmiss,y_nearmiss)


# If you look at the Learning curve for Logestic Regression above.. the train and CV curve run in (almost) parallel around 90% accuracy. Which means, providing more training data would not be of any help in improving the performance of the model.

# Lets run the SMOTE cross validation.
# NOTE - It takes ~ 10 mins for the code to run.

# In[ ]:


# from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

sm = SMOTETomek(random_state=42)
X_sm, y_sm = sm.fit_resample(undersample_X, undersample_y)

X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(X_sm, y_sm)

log_r.fit(X_train_sm,y_train_sm)
prediction_sm = log_r.predict(X_test_sm)
score_sm = accuracy_score(y_test_sm, prediction_sm)
print('Accuracy SMOTE',score_sm)
precision_score_sm = precision_score(y_test_sm, prediction_sm)
print('Precision Score SMOTE', precision_score_sm)
recall_score_sm = recall_score(y_test_sm, prediction_sm)
print('Recall Score - SMOTE', recall_score_sm)
f1_score_sm = f1_score(y_test_sm, prediction_sm)
print('F1 Score SMOTE',f1_score_sm)
roc_auc_score_sm = roc_auc_score(y_test_sm, prediction_sm)
print('ROC Score SMOTE',roc_auc_score_sm)


# In[ ]:


# it takes 67 mins to run this code.. So your Choice :)
# plot_learning_curve(log_r, X_sm,y_sm)


# Lets take a look at the Classification report and the Confusion Matrix for NearMiss and SMTOE cross validations

# In[ ]:


from sklearn.metrics import classification_report
# print('Classification Report for SMTOE',classification_report(y_test_sm, prediction_sm)) 
# print('Confusion Matrix for SMTOE-LogR',confusion_matrix(y_test_sm, prediction_sm))


# In[ ]:


print('Classification Report - NearMiss',classification_report(y_test_nm, prediction)) 
print('Confusion Matrix - NearMiss',confusion_matrix(y_test_nm, prediction))


# I would like to see if the Logistic regression is really the best model when we use NearMiss CV.. Lets find out..

# In[ ]:


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
models = []
models.append(('lreg', LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l1',
                   random_state=None, solver='warn', tol=0.0001, verbose=0,
                   warm_start=False)))
models.append(('knn', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=7, p=2, weights='uniform')))
models.append(('svc', SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
                         decision_function_shape='ovr', degree=3, gamma=0.1, kernel='linear',
                         max_iter=-1, probability=False, random_state=None, shrinking=True,
                         tol=0.001, verbose=False)))
models.append(('dtc', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')))

for name, model in models:
    model.fit(X_train_nm,y_train_nm)
    prediction = model.predict(X_test_nm)
    score = accuracy_score(y_test_nm, prediction)
    print('{} Accuracy NearMiss{}'.format(name,score_nm))
    precision_score_nm = precision_score(y_test_nm, prediction)
    print('{} Precision Score NearMiss{}'.format(name, precision_score_nm))
    recall_score_nm = recall_score(y_test_nm, prediction)
    print('{} Recall Score - NearMiss{}'.format(name,recall_score_nm))
    f1_score_nm = f1_score(y_test_nm, prediction)
    print('{} F1 Score NearMiss{}'.format(name, f1_score_nm))
    roc_auc_score_nm = roc_auc_score(y_test_nm, prediction)
    print('{} ROC Score NearMiss{}'.format(name, roc_auc_score_nm))
    fpr,tpr,threshold = roc_curve(y_test_nm, log_r.predict_proba(X_test_nm)[:,1])
    plt.figure(figsize = (8, 8))
    plt.plot(fpr, tpr, label = 'Model (area = %.2f)' % roc_auc_score_nm, color = 'r')
    plt.plot([0,1], [0,1], 'r--')
    plt.title('ROC Curve -- {}'.format(name))
    plt.xlabel('False Positive rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc = 'best')
    plt.show()
    
    plot_learning_curve(model, X_nearmiss,y_nearmiss)


# Hmm the Recall score for Decision tree Classifier is gone Up.. good news.. I'll take it. But wait, did you observe the DTC learining curve dipping down as the CV test size increases..
# So, what do you'll say, which model is a good model??
# Comments?

# In[ ]:




