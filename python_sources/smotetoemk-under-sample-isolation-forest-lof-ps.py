#!/usr/bin/env python
# coding: utf-8

# # Introduction of this Notebook
# 
# Although Credit Card Fraud Detection is already being explored in a number of ways, But here, I would like to represent the compile form of different algorithms. 

# # Content: 
# 1. Import Libraires and looking into the data
# 2. Exploratory Data Analysis
# 3. ML Algorithms
#     * Logistic Regression
#         * Full dataset
#         * Under-Sampling
#         * Over-Sampling
#             * SMOTETomek
#             * RandomOverSampler
#     * Isolation Forest
#     * Local Outlier Factor(LOF)

# ### 1. Import Libraires and looking into the data

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/creditcardfraud/creditcard.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.columns


# In[ ]:


df.describe().transpose()


# In[ ]:


df.shape


# In[ ]:


sns.heatmap(df.isnull(), yticklabels= False, cbar=False, cmap = 'viridis')


# This data does not have any null value hance saved a lot of time in data pre-processing.

# ### 2. Exploratory Data Analysis 

# In[ ]:


df.hist(figsize = (20,20) )
plt.show()


# In[ ]:


# Evaluating the percentage of fraud cases in the dataset

Fraud = df[df['Class'] == 1]
Normal = df[df['Class'] == 0]

Fraud_percentage = (len(Fraud)/len(Normal)) * 100
print(Fraud_percentage,  '%')
print('Fraud Cases: {}'.format(len(Fraud)))
print('Normal Cases: {}'.format(len(Normal)))


# In[ ]:


sns.countplot('Class', data = df, palette = ['b', 'r'])


# In[ ]:


# Correlation 
corr = df.corr()
sns.set_context('notebook', font_scale=1.0, rc = {'lines.linewidth': 2.5})
plt.figure(figsize = (20,20))

# mask the duplicate correlation values
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True

a = sns.heatmap(corr, mask = mask, annot=True, fmt = '.2f', cmap = 'viridis')

rotx = a.set_xticklabels(a.get_xticklabels(), rotation = 90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation = 30)


# In[ ]:


fig= plt.figure(figsize =(10,10))
ax1 = plt.subplot(2, 1, 1)
sns.scatterplot(x = 'Time', y = 'Amount', data =Fraud, ax = ax1)

ax2 = plt.subplot(2, 1, 2)
sns.scatterplot(x = 'Time', y = 'Amount', data =Normal, ax = ax2)


# ### 3. ML Algorithms
# * Logistic Regression
#     * Full dataset
#     * Under-Sampling
#     * Over-Sampling
#         *SMOTETomek
#         *RandomOverSampler
# * Isolation Forest
# * Local Outlier Factor(LOF)

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = df.drop('Class', axis = 1)
y = df.Class
X.shape, y.shape


# ### 1. Logistic Regression on Complete dataset

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression()
lgr.fit(X_train, y_train)
y_pred = lgr.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[ ]:


print('Classification report:')
print(classification_report(y_test, y_pred))
print('Confusion Metrix: ')
print(confusion_matrix(y_test, y_pred))
print('Accuracy Score')
print(accuracy_score(y_test, y_pred))


# **Note:**  Confusion metrix and accuracy score are not correct method to check the accuracy of the model, it is recommended to use AUC curve. 

# In[ ]:


from sklearn.metrics import roc_auc_score, roc_curve


# In[ ]:


y_score = lgr.predict_proba(X_test)[:,1]


# In[ ]:


FPR, TPR, threshold = roc_curve(y_test, y_score )

#Plot ROC curve
plt.title('ROC Curve of Logistic regression (full data)')
plt.plot(FPR, TPR)
plt.plot([0,1], ls ='--')
plt.plot([0,0], [1,0], c='.7'), plt.plot([1,1], c='.7')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


ras_lgr = roc_auc_score(y_test, y_score)
print(ras_lgr)


# ### 2. Under-sampling

# In[ ]:


from imblearn.under_sampling import NearMiss


# In[ ]:


# Implementing Undersampling 
nm = NearMiss()
X_us, y_us= nm.fit_sample(X,y)


# In[ ]:


X_us.shape, y_us.shape


# In[ ]:


from collections import Counter

print('Original dataset shape {}'.format(Counter(y)))
print('Undersampled dataset shape {}' .format(Counter(y_us)))


# In[ ]:


X_train_us, X_test_us, y_train_us, y_test_us = train_test_split(X_us, y_us, test_size=0.33, random_state=42)


# In[ ]:


X_train_us.shape, y_train_us.shape, X_test_us.shape, y_test_us.shape


# In[ ]:


lgr.fit(X_train_us, y_train_us)
y_pred_us = lgr.predict(X_test_us)


# In[ ]:


print('Classification report:')
print(classification_report(y_test_us, y_pred_us))
print('Confusion Metrix: ')
print(confusion_matrix(y_test_us, y_pred_us))
print('Accuracy Score')
print(accuracy_score(y_test_us, y_pred_us))


# In[ ]:


y_score_us = lgr.predict_proba(X_test_us)[:,1]


# In[ ]:


FPR, TPR, threshold = roc_curve(y_test_us, y_score_us )

#Plot ROC curve
plt.title('ROC Curve of undersampled')
plt.plot(FPR, TPR)
plt.plot([0,1], ls ='--')
plt.plot([0,0], [1,0], c='.7'), plt.plot([1,1], c='.7')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


ras_us = roc_auc_score(y_test_us, y_score_us)
print(ras_us)


# ### 3.  Over-Sampling
# Two methods:
# * SMOTETomek
# * RandomOverSampler 

# ### SMOTETomek

# In[ ]:


from imblearn.combine import SMOTETomek

smk = SMOTETomek(random_state = 42)
X_ov_smk, y_ov_smk = smk.fit_sample(X, y)


# In[ ]:


X_ov_smk.shape, y_ov_smk.shape


# In[ ]:


X_train_smk, X_test_smk, y_train_smk, y_test_smk = train_test_split(X_ov_smk, y_ov_smk, test_size=0.33, random_state=42)


# In[ ]:


X_train_smk.shape, y_train_smk.shape, X_test_smk.shape, y_test_smk.shape


# In[ ]:


lgr.fit(X_train_smk, y_train_smk)
y_pred_smk = lgr.predict(X_test_smk)


# In[ ]:


print('Classification report:')
print(classification_report(y_test_smk, y_pred_smk))
print('Confusion Metrix: ')
print(confusion_matrix(y_test_smk, y_pred_smk))
print('Accuracy Score')
print(accuracy_score(y_test_smk, y_pred_smk))


# In[ ]:


y_score_smk = lgr.predict_proba(X_test_smk)[:,1]


# In[ ]:


FPR, TPR, threshold = roc_curve(y_test_smk, y_score_smk )

#Plot ROC curve
plt.title('ROC Curve of Oversampled (SMOTETomek)')
plt.plot(FPR, TPR)
plt.plot([0,1], ls ='--')
plt.plot([0,0], [1,0], c='.7'), plt.plot([1,1], c='.7')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


ras_smk = roc_auc_score(y_test_smk, y_score_smk)
print(ras_smk)


# ### RandomOverSampler

# In[ ]:


# RandomOverSampler
from imblearn.over_sampling import RandomOverSampler


# In[ ]:


ros = RandomOverSampler()
X_ros, y_ros = ros.fit_sample(X, y)


# In[ ]:


X_ros.shape, y_ros.shape


# In[ ]:


X_train_ros, X_test_ros, y_train_ros, y_test_ros = train_test_split(X_ros, y_ros, test_size=0.33, random_state=42)


# In[ ]:


X_train_ros.shape, y_train_ros.shape, X_test_ros.shape, y_test_ros.shape


# In[ ]:


lgr.fit(X_train_ros, y_train_ros)
y_pred_ros = lgr.predict(X_test_ros)


# In[ ]:


print('Classification report:')
print(classification_report(y_test_ros, y_pred_ros))
print('Confusion Metrix: ')
print(confusion_matrix(y_test_ros, y_pred_ros))
print('Accuracy Score')
print(accuracy_score(y_test_ros, y_pred_ros))


# In[ ]:


y_score_ros = lgr.predict_proba(X_test_ros)[:,1]


# In[ ]:


FPR, TPR, threshold = roc_curve(y_test_ros, y_score_ros )

#Plot ROC curve
plt.title('ROC Curve of Oversampled (ROS)')
plt.plot(FPR, TPR)
plt.plot([0,1], ls ='--')
plt.plot([0,0], [1,0], c='.7'), plt.plot([1,1], c='.7')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


ras_ros = roc_auc_score(y_test_ros, y_score_ros)
print(ras_ros)


# # Unsupervised Outlier Detection
# * Isolation Forest Algorithm
# * Local Outlier Factor(LOF)

# ### 4.  Isolation Forest Algorithm

# In[ ]:


from sklearn.ensemble import IsolationForest


# In[ ]:


outlier_fraction = len(Fraud)/float(len(Normal))
n_outlier = len(Fraud)


# In[ ]:


isof = IsolationForest(max_samples=len(X),
                      contamination= outlier_fraction, 
                      random_state = 42)


# In[ ]:


isof.fit(X)
y_score_isof = isof.decision_function(X)
y_pred_isof = isof.predict(X)

# Reshape the prediction values to 0 for valid, 1 for fraud
y_pred_isof[y_pred_isof == 1] = 0
y_pred_isof[y_pred_isof == -1] = 1

n_errors = (y_pred_isof != y).sum()


# In[ ]:


print('Isolation Forest num of errors:' , n_errors)
print("Accuracy score of Isolation Forest is: ", accuracy_score(y, y_pred_isof))
print(classification_report(y, y_pred_isof))
print(confusion_matrix(y, y_pred_isof))


# In[ ]:


y_pred_isof_score = isof.score_samples(X)

fpr, tpr, thresholds = roc_curve(y, y_pred_isof)
plt.title('ROC Curve of Isolation Forest')
plt.plot(fpr, tpr)
plt.plot([0,1], ls ='--')
plt.plot([0,0], [1,0], c='.7'), plt.plot([1,1], c='.7')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# In[ ]:


ras_isof = roc_auc_score(y, y_pred_isof) 
print(ras_isof)


# ### 5.   Local Outlier Factor(LOF)

# In[ ]:


from sklearn.neighbors import LocalOutlierFactor

LOF = LocalOutlierFactor(n_neighbors = 20, 
                         algorithm = 'auto', 
                         leaf_size = 30,
                        metric = 'minkowski',
                        p = 2, metric_params = None, 
                        contamination = outlier_fraction)


# In[ ]:


y_pred_lof = LOF.fit_predict(X)
y_score_lof = LOF.negative_outlier_factor_

# Reshape the prediction values to 0 for valid, 1 for fraud
y_pred_lof[y_pred_lof == 1] = 0
y_pred_lof[y_pred_lof == -1] = 1

n_errors = (y_pred_lof != y).sum()

print(' Num of errors in LOF:' , n_errors)
print("Accuracy score of LOF is: ", accuracy_score(y, y_pred_lof))
print(classification_report(y, y_pred_lof))
print(confusion_matrix(y, y_pred_lof))


# In[ ]:


fpr, tpr, thresholds = roc_curve(y, y_pred_lof)
plt.title('ROC Curve of LOF')
plt.plot(fpr, tpr)
plt.plot([0,1], ls ='--')
plt.plot([0,0], [1,0], c='.7'), plt.plot([1,1], c='.7')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


ras_lof = roc_auc_score(y, y_pred_lof)
print(ras_lof)


# In[ ]:


print('ROC-AUC-SCORE of all the algorithms: ')
print('--' * 20)
print('Logistic Regression on Complete Data: ', ras_lgr)
print('Logistic Regression on Under-Sampled Data: ', ras_us)
print('Logistic Regression on Over-sampled (SMOTETomek): ', ras_smk)
print('Logistic Regression on Over-sampled (Randomoversampler) ', ras_ros)
print('Isolation Forest: ', ras_isof)
print('Local Outlier Factor (LOF): ', ras_lof)


# **Observations**   
# * Logestic regression on over-sampling (SMOTETomek) produce better result as compared to other. 

# Please comment and upvote, If you find it useful. 
