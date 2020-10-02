#!/usr/bin/env python
# coding: utf-8

# ## Credit Card Fraud Analysis
# 
# <s>I made this sometime ago with no real end goal in mind so I am going to go through the data a little and see where it takes me.</s>
# 
# After some further thought I have decied that the best way to look at this is to do some EDA and then focus on the sampling and prediction and how those things relate with various modeling methods such as:
# 
# - Linear Regression
# - Logit Regression
# - Random Forest (Maybe)
# - Light GBM

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, log_loss, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import statsmodels.api as sma
import lightgbm as lgb
from collections import Counter

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


df = pd.read_csv("../input/creditcard.csv", index_col=0)
# print(df.columns)
print(df.head(3))


# In[ ]:


df.info()


# In[ ]:


df.describe()


# We have a dataframe about 284000 entries and they are floats data types. The data is said to have been transformed via PCA in order to protect the information so we cant be sure what any of these categories really mean. Additonally, they are also standardized with the exception of the amount. 

# In[ ]:


Counter(df.Class)


# Class is a binary target with 492 instances of known fraud and 284,315 instances of non-fraud. Thats a ratio of:

# In[ ]:


print("Fraud to NonFraud Ratio of {:.3f}%".format(492/284315*100))


# Just from this we can see that there is a very low occurance of fraud in comparison with non fraud which could cause some issues should we dive into any kind of predictions.
# 
# Random Forest is supposed to be able to deal with ill-ratio'd samples fairly well but even this is pretty low. Some points I have learned about are __Undersampling__ and __Oversampling__ to _correct_ the proportions.

# In[ ]:


fig, ax=plt.subplots(1,1,figsize=(12,6))
df[df.Class==1]['Amount'].hist(bins=100,ax=ax,color='b');
plt.title('Histogram of Fraud Amounts');
plt.ylabel('Counts'); plt.xlabel('$');


# Looks like there a lot more instances of small fraud amounts than really large ones. 
# 
# This could be a sign of importance of _discreteness_ in the transaction amount. Maybe hoping that any actual fraud goes unnoticed in the account that is being fruaded.

# In[ ]:


fig, ax=plt.subplots(1,1,figsize=(10,8))
sns.heatmap(df.corr(), vmin=-1, vmax=1, ax=ax, cmap='coolwarm');
plt.title('Heat Map of Variable Correlations');


# Taking a glance at the variables we can see there is little to no correlation within the PCA variables (which should pretty much be expected) but there does seem to be some higher posititve correlation with __Amount__ and __V7__ and __Amount__ and __V20__ and some _higher_ negative correlation with __Amount__ and __V2__ and __V5__. 

# In[ ]:


corrs_amt = df.drop('Class',axis=1).corr()['Amount']
print(corrs_amt[np.abs(corrs_amt) > 0.3])


# In[ ]:


vars_to_cover = ['Amount','V2','V5','V7','V20']
print(df[vars_to_cover].corr())


# Glancing at the first row or column we can see that those correlations from the plot range from 0.33 to 0.53.

# Lets see what a simple linear regression on these variables looks like against the __AMOUNT__ field.

# In[ ]:


lin_mod = sma.OLS(exog=df[['V2','V5','V7','V20']], endog=df[['Amount']])
lin_fit = lin_mod.fit()
print(lin_fit.summary())
lin_pred = lin_fit.predict()
lin_pred_df = pd.DataFrame(lin_pred, index=df.index)


# In[ ]:


fig, ax=plt.subplots(1,1,figsize=(12,6))
lin_pred_df.iloc[:200].plot(ax=ax, style=['r-'], label='Pred', legend=True);
df.Amount.iloc[:200].plot(ax=ax, style=['b:'], label='Actuals', legend=True);
plt.title('First 200 Instances of Amount');
plt.xlabel('Time'); plt.ylabel('$');


# A simple linear regression model on the amounts highlight good connection with large fraud amounts. Its obvious that those variables alone dont take everything into account as far as predicting the amount goes. They do seem to notice some of the higher spikes but the model predicts __negative__ fraud which is __not__ something that would happen.

# ### Since the data set is fraudulent transactions for many different accounts, there is no need to worry too much about the time component since the accounts should for the most part be independent of each other.
# ### If it was multiple time points for multiple accounts then we would have to dive into other statistical features and nuances. 
# 
# We will just assume they are independent of each other.

# ## Logit Regression
# 
# Can we make some predictions on occurence of fraud based on the __Class__ variable of the data set?
# 
# First, lets take a look at our sampling issues.

# ### Oversampling
# This is where one would create observations in the data set belonging to the class that has the lower occurence.
# 
# Lets do this using the `imblearn` package and _SMOTE_ or Syntheitc Minority Over-Sampling Technique.
# 
# 

# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:


y_full = df['Class']
x_full = df.drop(['Class','Amount'], axis=1)


# In[ ]:


ism = SMOTE(random_state=42)


# In[ ]:


x_rs, y_rs = ism.fit_sample(x_full, y_full)
print('Resampled dataset shape {}'.format(Counter(y_rs)))


# In[ ]:


x_rs = pd.DataFrame(x_rs, columns = x_full.columns)
y_rs = pd.DataFrame(y_rs)


# This is an even balance. Using this information lets take a look at a simple logistic regression using some of the higher correlated variables with Class.

# In[ ]:


corrs = df.drop('Amount',axis=1).corr()['Class']
print(corrs[np.abs(corrs) > 0.2])


# Just taking a quick glance above, lets set a threshold above the abs(0.2) which leaves us with: __V10, V12, V14, & V17__.

# In[ ]:


xto, xvo, yto, yvo = train_test_split(x_rs, y_rs, test_size=0.2, random_state=42)
print(xto.shape, xvo.shape, yto.shape, yvo.shape)


# In[ ]:


from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.model_selection import StratifiedKFold


# In[ ]:


lr_os_mod = sma.Logit(endog = yto, exog = sma.add_constant(pd.DataFrame(xto, columns=x_rs.columns)[['V10','V12','V14','V17']]))
lr_os_fit = lr_os_mod.fit()
print(lr_os_fit.summary2())


# In[ ]:


lr_os_pred = lr_os_fit.predict(sma.add_constant(pd.DataFrame(xvo, columns=x_rs.columns)[['V10','V12','V14','V17']]))


# In[ ]:


print(lr_os_pred.head(), '\n', yvo.head())


# In order to get the predictions with in the realm of 0 and 1 we will simply round out at 0.5 and convert it to an __INT__ since the model gave us probabillites.

# In[ ]:


lr_os_pred_rnd = lr_os_pred.round(0).astype(int)
lr_os_pred_rnd.head()


# In[ ]:


confusion_matrix(lr_os_pred_rnd, yvo)/len(yvo)


# In[ ]:


print("Precision     : {:.4f}".format(precision_score(lr_os_pred_rnd, yvo)))
print("Recall        : {:.4f}".format(recall_score(lr_os_pred_rnd, yvo)))
print("Accuracy      : {:.4f}".format(accuracy_score(lr_os_pred_rnd, yvo)))
print("ROC/AUC Score : {:.4f}".format(roc_auc_score(lr_os_pred_rnd, yvo)))
print("F1 Score      : {:.4f}".format( 2*(precision_score(lr_os_pred_rnd, yvo)*recall_score(lr_os_pred_rnd, yvo))                                         / (precision_score(lr_os_pred_rnd, yvo)+recall_score(lr_os_pred_rnd, yvo)) ))


# A precision of 0.88 states that we got more relevant results than irrelevant and the high recall means we got most of the relevant results.
# 
# All of the scores for this model are fairly decent.

# ### UNDER SAMPLING
# 
# Random Under Sampling reduces the number of samples all togehter by randomly selecting a handful of samples from the class that is OVER-REPRESENTED.
# 

# In[ ]:


from imblearn.under_sampling import RandomUnderSampler


# In[ ]:


rus = RandomUnderSampler(random_state=42)


# In[ ]:


x_rs, y_rs = rus.fit_sample(x_full, y_full)
print('Resampled dataset shape {}'.format(Counter(y_rs)))


# In[ ]:


x_rs = pd.DataFrame(x_rs, columns = x_full.columns)
y_rs = pd.DataFrame(y_rs)


# With such a small data set, we will stick with the same regression variables we chose above.

# In[ ]:


xto, xvo, yto, yvo = train_test_split(x_rs, y_rs, test_size=0.2, random_state=42)
print(xto.shape, xvo.shape, yto.shape, yvo.shape)


# In[ ]:


lr_us_mod = sma.Logit(endog = yto, exog = sma.add_constant(pd.DataFrame(xto, columns=x_rs.columns)[['V10','V12','V14','V17']]))
lr_us_fit = lr_us_mod.fit()
print(lr_us_fit.summary2())


# Here we have seen that the significance of some of our variables has significantly decreased. In other words, __V17__ is not necessarily as good an indicator of FRAUD against such a smaller set of data. The p-value states that less that 65% of our samples (if randomly selected) would be with in the error thresholds of the model. Under normal circumstances we might drop this variable from the regression all together but we will leave it here.

# In[ ]:


lr_us_pred = lr_us_fit.predict(sma.add_constant(pd.DataFrame(xvo, columns=x_rs.columns)[['V10','V12','V14','V17']]))
lr_us_pred_rnd = lr_us_pred.round(0).astype(int)


# In[ ]:


print("Precision     : {:.4f}".format(precision_score(lr_us_pred_rnd, yvo)))
print("Recall        : {:.4f}".format(recall_score(lr_us_pred_rnd, yvo)))
print("Accuracy      : {:.4f}".format(accuracy_score(lr_us_pred_rnd, yvo)))
print("ROC/AUC Score : {:.4f}".format(roc_auc_score(lr_us_pred_rnd, yvo)))
print("F1 Score      : {:.4f}".format( 2*(precision_score(lr_us_pred_rnd, yvo)*recall_score(lr_us_pred_rnd, yvo))                                         / (precision_score(lr_us_pred_rnd, yvo)+recall_score(lr_us_pred_rnd, yvo)) ))


# All of our fields have gone down - likely due to the smaller data set however, most of these are still fairly decent.

# ### COMBINED SAMPLING
# 
# SMOTEENN - This is a method of over sampling using SMOTE and then tidying up the data set using ENN or _Edited Nearest Neighbors_. 
# 
# The ENN algorithm cleans up the over-sampled space by fine tuning the created samples and if necessary dropping ones that are outside of the calculated thresholds.

# In[ ]:


from imblearn.combine import SMOTEENN


# In[ ]:


cse = SMOTEENN(random_state=42)


# In[ ]:


x_rs, y_rs = cse.fit_sample(x_full, y_full)
print('Resampled dataset shape {}'.format(Counter(y_rs)))


# In[ ]:


x_rs = pd.DataFrame(x_rs, columns = x_full.columns)
y_rs = pd.DataFrame(y_rs)

# xto, xvo, yto, yvo = train_test_split(x_rs, y_rs, test_size=0.2, random_state=42)
# print(xto.shape, xvo.shape, yto.shape, yvo.shape)


# ### Light GBM 
# 
# Lets take another glance at this but from a different persepective. 
# 
# What if we wanted to narrow down the variables from the original 28 and then built a model off of that? 
# 
# ** I could do this with a Random Forest but I like Light GBM so Im using that. **

# In[ ]:


params = {'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'learning_rate': 0.05, 
    'max_depth': 5,
    'num_leaves': 92, 
    'min_data_in_leaf': 46, 
    'lambda_l1': 1.0,
    'feature_fraction': 0.8, 
    'bagging_fraction': 0.8, 
    'bagging_freq': 5}

kfolds = 3
kd = 0
preds = 0
for i in range(kfolds):
    print('In kfold:',str(i+1))
    xt,xv,yt,yv = train_test_split(x_rs, y_rs, test_size=0.2, random_state=(i*42))
    
    trn = lgb.Dataset(xt,yt.values.flatten())
    val = lgb.Dataset(xv,yv.values.flatten())
    model = lgb.train(params, train_set=trn, num_boost_round=100,
                     valid_sets=[val], valid_names=['val'],
                     verbose_eval=20,
                     early_stopping_rounds=40)
    
    pred = model.predict(xv, num_iteration=model.best_iteration+50)
    preds += pred
    kd += 1
    print('=========================')
    print("    Precision : {:.4f}".format(precision_score(np.round(pred,0).astype(int), yv)))
    print("    Recall    : {:.4f}".format(recall_score(np.round(pred,0).astype(int), yv)))
    print("    Accuracy  : {:.4f}".format(accuracy_score(np.round(pred,0).astype(int), yv)))
    print("ROC/AUC Score : {:.4f}".format(roc_auc_score(np.round(pred,0).astype(int), yv)))
    print("    F1 Score  : {:.4f}".format( 2*(precision_score(np.round(pred,0).astype(int), yv)*recall_score(np.round(pred,0).astype(int), yv))                                         / (precision_score(np.round(pred,0).astype(int), yv)+recall_score(np.round(pred,0).astype(int), yv)) ))
    print('=========================')
preds /= kd


# In[ ]:


lgb.plot_importance(model, figsize=(12,8));


# In[ ]:


X = x_rs[['V4','V10','V12','V14']]
y = y_rs
logmod = sma.Logit(endog=y, exog=sma.add_constant(X))
logfit = logmod.fit()
print(logfit.summary2())


# In[ ]:


logPred = logfit.predict(sma.add_constant(x_rs[['V4','V10','V12','V14']]))
print(logPred.head(3))


# In[ ]:


print("Precision     : {:.4f}".format(precision_score(np.round(logPred,0).astype(int), y_rs)))
print("Recall        : {:.4f}".format(recall_score(np.round(logPred,0).astype(int), y_rs)))
print("Accuracy      : {:.4f}".format(accuracy_score(np.round(logPred,0).astype(int), y_rs)))
print("ROC/AUC Score : {:.4f}".format(roc_auc_score(np.round(logPred,0).astype(int), y_rs)))
print("F1 Score      : {:.4f}".format( 2*(precision_score(np.round(logPred,0).astype(int), y_rs)*recall_score(np.round(logPred,0).astype(int), y_rs))                                     / (precision_score(np.round(logPred,0).astype(int), y_rs)+recall_score(np.round(logPred,0).astype(int), y_rs)) ))


# In[ ]:


print(lr_os_pred.shape, lr_us_pred.shape, logPred.shape, y_rs.shape)


# We cant use the same items from before due to the differing sizes so we will just use the predict methods from each one.

# In[ ]:


pred_methods = [lr_os_fit.predict(sma.add_constant(pd.DataFrame(x_rs, columns=x_rs.columns)[['V10','V12','V14','V17']])),
                lr_us_fit.predict(sma.add_constant(pd.DataFrame(x_rs, columns=x_rs.columns)[['V10','V12','V14','V17']])), 
                logfit.predict(sma.add_constant(x_rs[['V4','V10','V12','V14']]))]


# In[ ]:


cols = ['b','r','g','m','c']
pred_fits = ['OverSampling - Logit', 'UnderSampling - Logit', 'Combined Sampling - Light GBM']
fig, ax=plt.subplots(1,1,figsize=(16,10))
plt.title('Variations on Sampling & Models ROC/AUC Curves');
for i in range(len(pred_methods)):
    fpr, tpr, thresholds = roc_curve(y_rs, pred_methods[i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=1, alpha=0.9, color=cols[i],
                 label='Model: %s   (AUC = %0.4f)' % (pred_fits[i], roc_auc))
ax.plot(np.linspace(0,1,100),np.linspace(0,1,100), 'k:');
ax.legend();
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate');


# We werent extremely rigorous in testing all of our models on each and set of data and we did not full tune our Light GBM model, however its clear that further gains can be made if we spend a little time on these items.
# 
# For other, wider problems, where we dont have the luxury of only a few features, the light GBM method would likely be much more efficient in correctly predicting.
# 
# In summary, depending on the model type and the amount of data you have, it is necessary to be vigilant in keeping your sample ratios in mind and your sampling methods in check.

# In[ ]:


# cv = StratifiedKFold(n_splits=5)
# mod = LogisticRegression(C=5, fit_intercept=True, penalty='l2',
#                            n_jobs=1, verbose=20, random_state=42)


# In[ ]:


# X_lr = df.drop(['Class','Amount'],axis=1).values
# y_lr = df['Class'].values


# In[ ]:


# tprs = []
# aucs = []
# i=0
# cols = ['b','r','g','m','c']
# fig, ax=plt.subplots(1,1,figsize=(12,6))
# plt.title('LogReg ROC AUC Curve');
# for trn, tst in cv.split(X_lr,y_lr):
#     probs = mod.fit(X_lr[trn],y_lr[trn]).predict_proba(X_lr[tst])
#     fpr, tpr, thresholds = roc_curve(y_lr[tst], probs[:, 1])
#     roc_auc = auc(fpr, tpr)
#     aucs.append(roc_auc)
#     ax.plot(fpr, tpr, lw=1, alpha=0.9, color=cols[i],
#              label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
#     i+=1
# ax.plot(np.linspace(0,1,100),np.linspace(0,1,100), 'k:');
# ax.legend();
# plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate');


# In[ ]:




