#!/usr/bin/env python
# coding: utf-8

# # Unbalanced Classification for Fraud Detection
# 
# ## Objective
# 
# The input dataset contains features such as credit card transaction time, amount, and other hidden features.
# The outcome is class (fraud or not).
# 
# Using machine learning, we will predict whether a transaction is fraud or not.
# 
# In particular, we need to deal with an imbalanced dataset. To do so, we consider these approaches:
# - Collecting more data, if possible
# - Changing the performance metrics (precision, recall, f1-score, ...)
# - Resampling the data (oversampling, undersampling, SMOTE, ...)
# - Changing the MLAs
# 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn import metrics
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# ## Data Cleaning

# In[ ]:


df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.info()


# No gap or missing value exists in any variable.

# In[ ]:


df.describe()


# For the Amount and Class, the data description makes sense, and the mean, min, and max values are reasonable. For example, there is no negative value.

# ### No Feature Engineering:
# 
# To keep the privacy of customers, the input data data was scaled and the feature names were hidden. So, we have no information about the nature of features, and feature engineering is not practical in this project.

# ## Exploratory Data Analysis

# In[ ]:


fraud = df.Class[df.Class == 1].count() / df.Class.count() * 100
non_fraud = 100 - fraud
print('% of fraud transactions: ', fraud)
print('% of non-fraud transactions: ', non_fraud)

sb.countplot('Class', data=df, palette='RdBu_r')
plt.show()


# In[ ]:


sb.boxplot(x = 'Class', y ='Amount', data = df, showfliers = False)
plt.show()


# There seems to be no significant difference in the amount of transaction between non-fraud and fraud transactions.

# In[ ]:


plt.figure(figsize = [10, 8])
sb.pairplot(df[['Time','Amount','Class']], hue='Class', palette = 'husl')
plt.show()


# ### Scaling Amount and Time:

# In[ ]:


df['Amount'] = RobustScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df['Time'] = RobustScaler().fit_transform(df['Time'].values.reshape(-1,1))


# ## Implementing MLAs:
# 
# ### Spliting the data into test and train sets:
# 
# - We use basic split method, which is "train_test_split".
# - Since the data is imbalanced, there is a risk that the ratio of fraud to non-fraud changes by using "train_test_split". What if there is no fraud case in test set?
# - To address this concern, we calculate the percentage of fraud and non-fraud cases in test and train sets. Fortunately, the percentage of classes is unchanged in test and train sets.
# - Alternatively, you can use stratified sampling.

# In[ ]:


y = df.Class
X = df.drop('Class', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


tr_fraud = y_train[y_train == 1].count() / y_train.count() * 100
tr_non_fraud = 100 - tr_fraud

print('% of train fraud transactions: ', tr_fraud)
print('% of train non-fraud transactions: ', tr_non_fraud)
print('\n')

te_fraud = y_test[y_test == 1].count() / y_test.count() * 100
te_non_fraud = 100 - te_fraud
print('% of test fraud transactions: ', te_fraud)
print('% of test non-fraud transactions: ', te_non_fraud)


# ### 1. Defining a baseline:
# 
# Here, the baseline is a naive model that always gives non-fraud as the answer!
# When we implement an MLA, our final result should be much better than the baseline; otherwise our model is useles.

# In[ ]:


y_baseline = y_test.copy()
y_baseline[:] = 0

print('Accuracy:', metrics.accuracy_score(y_test, y_baseline))
print('Recall:', metrics.recall_score(y_test, y_baseline))
print('Precision:', metrics.precision_score(y_test, y_baseline))
print('f1-score:', metrics.f1_score(y_test, y_baseline))


# ### 2. Implementing Logistic Regression:

# In[ ]:


log_reg = LogisticRegression(solver='liblinear')
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)


# ### Changing the performance metrics.
# 
# When working with an imbalanced data, accuracy is very misleading. Instead, we use recall, precision, and f1-score. 

# In[ ]:


print('Accuracy:', metrics.accuracy_score(y_test, y_pred_lr))
print('Recall:', metrics.recall_score(y_test, y_pred_lr))
print('Precision:', metrics.precision_score(y_test, y_pred_lr))
print('f1-score:', metrics.f1_score(y_test, y_pred_lr))


# The imbalanced dataset causes low recall, precision, and f1-score.

# ### 3. Improved Logistic Regression:
# 
# We will use the hyperparameter class_weight, so that our MLA will be more appropriate for imbalanced data.

# In[ ]:


log_reg_cw = LogisticRegression(solver='liblinear', class_weight='balanced')
log_reg_cw.fit(X_train, y_train)
y_pred_lr_cw = log_reg_cw.predict(X_test)

print('Accuracy:', metrics.accuracy_score(y_test, y_pred_lr_cw))
print('Recall:', metrics.recall_score(y_test, y_pred_lr_cw))
print('Precision:', metrics.precision_score(y_test, y_pred_lr_cw))
print('f1-score:', metrics.f1_score(y_test, y_pred_lr_cw))


# Recall is improved at the expense of lowering precision (which is close to zero); so this is not satisfactory.

# ### 4. Changing the MLAs (Here, Random Forest)
# 
# Some MLAs are not proper for imbalanced data (e.g. logistic regresison), but some others perform better on both balanced and imbalanced data (decision tree, and random forest).

# In[ ]:


rand_forst = RandomForestClassifier(n_estimators=50, random_state=0)
rand_forst.fit(X_train, y_train)
y_pred_rf = rand_forst.predict(X_test)


# In[ ]:


print('Accuracy:', metrics.accuracy_score(y_test, y_pred_rf))
print('Recall:', metrics.recall_score(y_test, y_pred_rf))
print('Precision:', metrics.precision_score(y_test, y_pred_rf))
print('f1-score:', metrics.f1_score(y_test, y_pred_rf))


# Although recall, precision, and f1-ssocre are not ideal, random forest provided the highest scores, so far.

# ### Confusion Matrix for Random Forest:

# In[ ]:


#y_pred = cross_val_predict(rand_forst, X_train, y_train)
sb.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, cmap='Greens')
plt.xlabel('Modeled counts')
plt.ylabel('True counts')
plt.show()


# - The result of confusion matrix is good when the values on the upper left and lower right (True negative and true positive, respectively) are high and other values are low, which is the case here.
# - This verifies that there is no systematic error that our MLA always predicts non-fraud when there is fraud.
# - False negative is not ideal (~ one fifth of all negatives), which is why recall is not too high.

# ## Resampling the data:
# 
# **Important:**
# 
# To avoid overfitting, we need to split the data into test and train sets BEFORE resampling. Resampling should be performed on train set, NOT test set.

# ### a) Oversampling:
# 
# - Oversampling increases the number of cases of minority class by adding copies of them to the dataset.
# - If you do not have a huge dataset (of the order of million rows), then oversampling might work for you.

# In[ ]:


df_train = pd.concat([X_train, y_train], axis=1)

fraid_tr = df_train[df_train.Class == 1]
non_fraid_tr = df_train[df_train.Class == 0]

fraud_tr_os = resample(fraid_tr, n_samples=len(non_fraid_tr), replace=True, random_state=0)

df_tr_os = pd.concat([fraud_tr_os, non_fraid_tr])


# In[ ]:


fraud = fraud_tr_os.Class.count() / df_tr_os.Class.count() * 100
non_fraud = 100 - fraud
print('% of fraud transactions: ', fraud)
print('% of non-fraud transactions: ', non_fraud)

sb.countplot('Class', data=df_tr_os, palette='hls')
plt.show()


# - The number of fraud cases are increased and is now equal to the number of non-fraud cases.
# - The balanced dataset shows higher correlation between some variables, compared to imbalanced dataset. For example, V4 and V11 show relatively high positive correlation with Class, whereas V12 and V14 show relatively high negative correlation with Class:

# In[ ]:


plt.figure(figsize = [10, 8])
sb.heatmap(df_tr_os.corr(), vmin=-1, vmax=1, cmap = 'RdBu_r') #annot=True
plt.show()


# In[ ]:


plt.figure(figsize = [10, 8])
sb.pairplot(df_tr_os[['V4','V11','V12','V14','Class']], hue='Class', palette = 'husl')
plt.show()


# We see that the outcome Class is strongly dependent of features such as V4, V11, V12, and V14.

# In[ ]:


fig, axis = plt.subplots(1, 3,figsize=(15,5))
sb.boxplot(x = 'Class', y ='V4', data = df_tr_os, ax = axis[0], showfliers = False, palette = 'hls')
sb.boxplot(x = 'Class', y ='V11', data = df_tr_os, ax = axis[1], showfliers = False, palette = 'RdBu_r')
sb.boxplot(x = 'Class', y ='V14', data = df_tr_os, ax = axis[2], showfliers = False)
plt.show()


# ### Logistic Regression after Oversampling:

# In[ ]:


X_tr = df_tr_os.drop('Class', axis=1)
y_tr = df_tr_os.Class

log_reg = LogisticRegression(solver='liblinear')
log_reg.fit(X_tr, y_tr)
y_pred_lr_os = log_reg.predict(X_test)

print('Accuracy:', metrics.accuracy_score(y_test, y_pred_lr_os))
print('Recall:', metrics.recall_score(y_test, y_pred_lr_os))
print('Precision:', metrics.precision_score(y_test, y_pred_lr_os))
print('f1-score:', metrics.f1_score(y_test, y_pred_lr_os))


# The oversampling has improved the recall significantly, but at the expense of lowering the precision.

# ### Random Forest after Oversampling:

# In[ ]:


rand_forst = RandomForestClassifier(n_estimators=50, random_state=0)
rand_forst.fit(X_tr, y_tr)
y_pred_rf_os = rand_forst.predict(X_test)

print('Accuracy:', metrics.accuracy_score(y_test, y_pred_rf_os))
print('Recall:', metrics.recall_score(y_test, y_pred_rf_os))
print('Precision:', metrics.precision_score(y_test, y_pred_rf_os))
print('f1-score:', metrics.f1_score(y_test, y_pred_rf_os))


# - Oversampling slightly improves all three scores (recall, precision, f1-score).
# - The confusion matrix shows that oversampling decreases the false negatives:

# In[ ]:


sb.heatmap(confusion_matrix(y_test, y_pred_rf_os), annot=True, cmap='RdPu')
plt.xlabel('Modeled counts')
plt.ylabel('True counts')
plt.show()


# ### b) Undersampling
# 
# - Another resampling method is undersampling, which removes some data from the majority class.
# - Pros: works well when we have a huge dataset (e.g. of the order of million rows).
# - cons: we are losing valuable informaiton by removing data points, and this might cause underfitting.
# - Therefore, we don't explore undersampling in this project.

# ### c) SMOTE
# 
# - SMOTE or Synthetic Minority Oversampling Technique uses the k-nearest neighbors technique to create synthetic and new data points.

# In[ ]:


smote = SMOTE(sampling_strategy=1.0, random_state=0)
X_train_sm, y_train_sm = smote.fit_sample(X_train, y_train)


# In[ ]:


fraud = y_train_sm[y_train_sm == 1].count() / y_train_sm.count() * 100
non_fraud = 100 - fraud
print('% of fraud transactions: ', fraud)
print('% of non-fraud transactions: ', non_fraud)


# ### Logistic Regression after SMOTE:

# In[ ]:


log_reg = LogisticRegression(solver='liblinear')
log_reg.fit(X_train_sm, y_train_sm)
y_pred_lr_sm = log_reg.predict(X_test)

print('Accuracy:', metrics.accuracy_score(y_test, y_pred_lr_sm))
print('Recall:', metrics.recall_score(y_test, y_pred_lr_sm))
print('Precision:', metrics.precision_score(y_test, y_pred_lr_sm))
print('f1-score:', metrics.f1_score(y_test, y_pred_lr_sm))


# Similar to oversampling, SMOTE has improved the recall significantly, but at the expense of lowering the precision.

# ### Random Forest after SMOTE:

# In[ ]:


rand_forst = RandomForestClassifier(n_estimators=50, random_state=0)
rand_forst.fit(X_train_sm, y_train_sm)
y_pred_rf_sm = rand_forst.predict(X_test)

print('Accuracy:', metrics.accuracy_score(y_test, y_pred_rf_sm))
print('Recall:', metrics.recall_score(y_test, y_pred_rf_sm))
print('Precision:', metrics.precision_score(y_test, y_pred_rf_sm))
print('f1-score:', metrics.f1_score(y_test, y_pred_rf_sm))


# - Compared to oversampling, SMOTE slightly improves recall, but precision slightly decreases.
# - Compared to oversampling, the confusion matrix shows that SMOTE slightly decreases the false negatives, but slightly increases false posivies.

# In[ ]:


sb.heatmap(confusion_matrix(y_test, y_pred_rf_sm), annot=True, cmap='Purples')
plt.xlabel('Modeled counts')
plt.ylabel('True counts')
plt.show()


# In[ ]:


print('Logistic Regression: ', roc_auc_score(y_test, y_pred_lr))
print('Improved Logistic Regression: ', roc_auc_score(y_test, y_pred_lr_cw))
print('Random Forest: ', roc_auc_score(y_test, y_pred_rf))
print('Logistic Regression after Oversampling: ', roc_auc_score(y_test, y_pred_lr_os))
print('Random Forest after Oversampling: ', roc_auc_score(y_test, y_pred_rf_os))
print('Logistic Regression after SMOTE: ', roc_auc_score(y_test, y_pred_lr_sm))
print('Random Forest after SMOTE: ', roc_auc_score(y_test, y_pred_rf_sm))


# ### Summary: ROC Curve

# In[ ]:


fpr_lr, tpr_lr, thr_lr          = roc_curve(y_test, y_pred_lr)
fpr_lr_os, tpr_lr_os, thr_lr_os = roc_curve(y_test, y_pred_lr_os)
fpr_lr_sm, tpr_lr_sm, thr_lr_sm = roc_curve(y_test, y_pred_lr_sm)
fpr_rf, tpr_rf, thr_rf          = roc_curve(y_test, y_pred_rf)
fpr_rf_os, tpr_rf_os, thr_rf_os = roc_curve(y_test, y_pred_rf_os)
fpr_rf_sm, tpr_rf_sm, thr_rf_sm = roc_curve(y_test, y_pred_rf_sm)

plt.figure(figsize=(10,8))
plt.plot(fpr_lr, tpr_lr,       label='Logistic regression')
plt.plot(fpr_lr_os, tpr_lr_os, label='Logistic regression after oversampling')
plt.plot(fpr_lr_sm, tpr_lr_sm, label='Logistic regression after SMOTE')
plt.plot(fpr_rf, tpr_rf,       label='Random forest')
plt.plot(fpr_rf_os, tpr_rf_os, label='Random forest after oversampling')
plt.plot(fpr_rf_sm, tpr_rf_sm, label='Random forest after SMOTE')
plt.plot([0, 1], [0, 1], 'k:')

plt.xlim([-0.05, 1])
plt.ylim([0, 1])
plt.legend(fontsize=12)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.show()


# - Among resampling methods, SMOTE leads to highest TPR.
# - Among MLAs, ramdom forest has lower FPR, but logistic regression has higher TPR.
# - The trade-off between higher TPR and lower FPR depends on the project objective. What percentage of false positives is acceptable? 
# - In the case of fraud detection, it is not desirable to increase true positive rate at the expense of increasing false positive rate. We do not want our MLA to falsely detect many transactions as fraud!
# - Therefore, I am willing to select random forest after SMOTE as the best MLA. Note that SMOTE method is computationally-expensive, so if you have a huge dataset, you might want to use random forest after oversampling or undersampling.

# In[ ]:




