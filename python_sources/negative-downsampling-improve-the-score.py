#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import Image
display(Image('../input/charlie/Ou est charlie.PNG', width= 800, unconfined=True))


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold,KFold
from bayes_opt import BayesianOptimization
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve
from sklearn import metrics
from sklearn import preprocessing
import itertools
from scipy import interp
# Lgbm
import lightgbm as lgb
import seaborn as sns


import matplotlib.pylab as plt


import os
import gc

import datetime

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


def Negativedownsampling(train, ratio) :
    

    # Number of data points in the minority class
    number_records_fraud = len(train[train.isFraud == 1])
    fraud_indices = np.array(train[train.isFraud == 1].index)

    # Picking the indices of the normal classes
    normal_indices = train[train.isFraud == 0].index

    # Out of the indices we picked, randomly select "x" number (number_records_fraud)
    random_normal_indices = np.random.choice(normal_indices, number_records_fraud*ratio, replace = False)
    random_normal_indices = np.array(random_normal_indices)

    # Appending the 2 indices
    under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

    # Under sample dataset
    under_sample_data = train.iloc[under_sample_indices,:]
    
    # Showing ratio
    print("Percentage of normal transactions: ", round(len(under_sample_data[under_sample_data.isFraud == 0])/len(under_sample_data),2)* 100,"%")
    print("Percentage of fraud transactions: ", round(len(under_sample_data[under_sample_data.isFraud == 1])/len(under_sample_data),2)* 100,"%")
    print("Total number of transactions in resampled data: ", len(under_sample_data))
    
    return under_sample_data

    


# In[ ]:


train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')
train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')
sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')


# In[ ]:


# merge 
df_train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
df_test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

print("Train shape : "+str(df_train.shape))
print("Test shape  : "+str(df_test.shape))


del train_transaction, train_identity, test_transaction, test_identity
gc.collect()


# In[ ]:


rm_cols = ['TransactionID', 'TransactionDT', 'isFraud']


# In[ ]:


features = []
features = [col for col in list(df_train) if col not in rm_cols]


# In[ ]:


for f in features:
    if(str(df_train[f].dtype)!="object" and str(df_train[f].dtype) !="category") :
        df_train[f] = df_train[f].replace(np.nan,-999)
        df_test[f] = df_test[f].replace(np.nan,-999)


# In[ ]:


# Label Encoding
for f in features:
    if  (str(df_train[f].dtype)=="object" or str(df_train[f].dtype)=="category") :  
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_train[f].values) + list(df_test[f].values))
        df_train[f] = lbl.transform(list(df_train[f].values))
        df_test[f] = lbl.transform(list(df_test[f].values))
df_train = df_train.reset_index()
df_test = df_test.reset_index()


# In[ ]:


df_train.sort_values('TransactionDT', inplace = True)


# In[ ]:


df_train_resampling_1 = Negativedownsampling(df_train, 9)


# In[ ]:


df_train_resampling_2 = Negativedownsampling(df_train, 3)


# In[ ]:


target = 'isFraud'


# # Confusion Matrix

# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion matrix"',
                          cmap = plt.cm.Blues, percentage = False) :
    
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    
    
    if (percentage == True) :

        plt.text(0, 0, str(round((cm[0, 0]/(cm[0, 0]+cm[0, 1]))*100,2))+" %",
                 horizontalalignment = 'center',
                 color = 'white' if cm[0, 0] > thresh else 'black')

        plt.text(1, 0, str(round((cm[0, 1]/(cm[0, 0]+cm[0, 1]))*100,2))+" %",
                 horizontalalignment = 'center',
                 color = 'white' if cm[0, 1] > thresh else 'black')

        plt.text(1, 1, str(round((cm[1,1]/(cm[1, 1]+cm[1, 0]))*100,2))+" %",
                 horizontalalignment = 'center',
                 color = 'white' if cm[1, 1] > thresh else 'black')

        plt.text(0, 1, str(round((cm[1, 0]/(cm[1, 1]+cm[1, 0]))*100,2)) +" %",
                 horizontalalignment = 'center',
                 color = 'white' if cm[1,0] > thresh else 'black')

    
    else :
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :
            plt.text(j, i, cm[i, j],
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


lgb_params = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 2**8,
    "learning_rate" : 0.1,
    "bagging_freq": 5,
    "bagging_fraction" : 0.4,
    "feature_fraction" : 1,
    "min_data_in_leaf": 80,
    "min_sum_heassian_in_leaf": 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    #"lambda_l1" : 5,
    #"lambda_l2" : 5,
    "bagging_seed" : 2019,
    "verbosity" : 1,
    "seed": 2019
}


# # Full Data :

# In[ ]:


plt.rcParams["axes.grid"] = True

nfold = 3
skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)

oof = np.zeros(len(df_train))
mean_fpr = np.linspace(0,1,100)
cms= []
tprs = []
aucs = []
y_real = []
y_proba = []
recalls = []
roc_aucs = []
f1_scores = []
accuracies = []
precisions = []
predictions = np.zeros(len(df_test))
feature_importance_df = pd.DataFrame()

i = 1
for train_idx, valid_idx in skf.split(df_train, df_train.isFraud.values):
    print("\nfold {}".format(i))
    trn_data = lgb.Dataset(df_train.iloc[train_idx][features].values,
                                   label=df_train.iloc[train_idx][target].values, feature_name=features 
                                   )
    val_data = lgb.Dataset(df_train.iloc[valid_idx][features].values,
                                   label=df_train.iloc[valid_idx][target].values,  feature_name=features
                                   )   
    
    clf = lgb.train(lgb_params, trn_data, num_boost_round = 500, valid_sets = [trn_data, val_data], verbose_eval = 250, early_stopping_rounds = 100)
    oof[valid_idx] = clf.predict(df_train.iloc[valid_idx][features].values) 
    
    predictions += clf.predict(df_test[features]) / nfold
    
    # Scores 
    roc_aucs.append(roc_auc_score(df_train.iloc[valid_idx][target].values, oof[valid_idx]))
    accuracies.append(accuracy_score(df_train.iloc[valid_idx][target].values, oof[valid_idx].round()))
    recalls.append(recall_score(df_train.iloc[valid_idx][target].values, oof[valid_idx].round()))
    precisions.append(precision_score(df_train.iloc[valid_idx][target].values ,oof[valid_idx].round()))
    f1_scores.append(f1_score(df_train.iloc[valid_idx][target].values, oof[valid_idx].round()))
    
    # Roc curve by folds
    f = plt.figure(1)
    fpr, tpr, t = roc_curve(df_train.iloc[valid_idx][target].values, oof[valid_idx])
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (i,roc_auc))
    
    # Precion recall by folds
    g = plt.figure(2)
    precision, recall, _ = precision_recall_curve(df_train.iloc[valid_idx][target].values, oof[valid_idx])
    y_real.append(df_train.iloc[valid_idx][target].values)
    y_proba.append(oof[valid_idx])
    plt.plot(recall, precision, lw=2, alpha=0.3, label='P|R fold %d' % (i))  
    
    i= i+1
    
    # Confusion matrix by folds
    cms.append(confusion_matrix(df_train.iloc[valid_idx][target].values, oof[valid_idx].round()))


# Metrics
print(
        '\nCV roc score        : {0:.4f}, std: {1:.4f}.'.format(np.mean(roc_aucs), np.std(roc_aucs)),
        '\nCV accuracy score   : {0:.4f}, std: {1:.4f}.'.format(np.mean(accuracies), np.std(accuracies)),
        '\nCV recall score     : {0:.4f}, std: {1:.4f}.'.format(np.mean(recalls), np.std(recalls)),
        '\nCV precision score  : {0:.4f}, std: {1:.4f}.'.format(np.mean(precisions), np.std(precisions)),
        '\nCV f1 score         : {0:.4f}, std: {1:.4f}.'.format(np.mean(f1_scores), np.std(f1_scores))
)

#ROC 
f = plt.figure(1)
plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'grey')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.4f)' % (np.mean(roc_aucs)),lw=2, alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LGB ROC curve by folds')
plt.legend(loc="lower right")

# PR plt
g = plt.figure(2)
plt.plot([0,1],[1,0],linestyle = '--',lw = 2,color = 'grey')
y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
precision, recall, _ = precision_recall_curve(y_real, y_proba)
plt.plot(recall, precision, color='blue',
         label=r'Mean P|R')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('P|R curve by folds')
plt.legend(loc="lower left")

# Confusion maxtrix & metrics
plt.rcParams["axes.grid"] = False
cm = np.average(cms, axis=0)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cm, 
                      classes=class_names, 
                      title= 'LGB Confusion matrix [averaged/folds]')
plt.show()

plt.figure()
plot_confusion_matrix(cm, 
                      classes=class_names, 
                      title= 'LGB Confusion matrix [averaged/folds] percentage',percentage = True)
plt.show()


# # Resampling 1 :

# In[ ]:


plt.rcParams["axes.grid"] = True


oof_resampling_1 = np.zeros(len(df_train_resampling_1))
mean_fpr = np.linspace(0,1,100)
cms= []
tprs = []
aucs = []
y_real = []
y_proba = []
recalls = []
roc_aucs = []
f1_scores = []
accuracies = []
precisions_resampling_1 = []
predictions_resampling_1 = np.zeros(len(df_test))
feature_importance_df = pd.DataFrame()

i = 1
for train_idx, valid_idx in skf.split(df_train_resampling_1, df_train_resampling_1.isFraud.values):
    print("\nfold {}".format(i))
    trn_data = lgb.Dataset(df_train_resampling_1.iloc[train_idx][features].values,
                                   label=df_train_resampling_1.iloc[train_idx][target].values, feature_name=features 
                                   )
    val_data = lgb.Dataset(df_train_resampling_1.iloc[valid_idx][features].values,
                                   label=df_train_resampling_1.iloc[valid_idx][target].values,  feature_name=features
                                   )   
    
    clf = lgb.train(lgb_params, trn_data, num_boost_round = 500, valid_sets = [trn_data, val_data], verbose_eval = 250, early_stopping_rounds = 100)
    oof_resampling_1[valid_idx] = clf.predict(df_train_resampling_1.iloc[valid_idx][features].values) 
    
    predictions_resampling_1 += clf.predict(df_test[features]) / nfold
    
    # Scores 
    roc_aucs.append(roc_auc_score(df_train_resampling_1.iloc[valid_idx][target].values, oof_resampling_1[valid_idx]))
    accuracies.append(accuracy_score(df_train_resampling_1.iloc[valid_idx][target].values, oof_resampling_1[valid_idx].round()))
    recalls.append(recall_score(df_train_resampling_1.iloc[valid_idx][target].values, oof_resampling_1[valid_idx].round()))
    precisions.append(precision_score(df_train_resampling_1.iloc[valid_idx][target].values ,oof_resampling_1[valid_idx].round()))
    f1_scores.append(f1_score(df_train_resampling_1.iloc[valid_idx][target].values, oof_resampling_1[valid_idx].round()))
    
    # Roc curve by folds
    f = plt.figure(1)
    fpr, tpr, t = roc_curve(df_train_resampling_1.iloc[valid_idx][target].values, oof_resampling_1[valid_idx])
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (i,roc_auc))
    
    # Precion recall by folds
    g = plt.figure(2)
    precision, recall, _ = precision_recall_curve(df_train_resampling_1.iloc[valid_idx][target].values, oof_resampling_1[valid_idx])
    y_real.append(df_train_resampling_1.iloc[valid_idx][target].values)
    y_proba.append(oof_resampling_1[valid_idx])
    plt.plot(recall, precision, lw=2, alpha=0.3, label='P|R fold %d' % (i))  
    
    i= i+1
    
    # Confusion matrix by folds
    cms.append(confusion_matrix(df_train_resampling_1.iloc[valid_idx][target].values, oof_resampling_1[valid_idx].round()))
    

# Metrics
print(
        '\nCV roc score        : {0:.4f}, std: {1:.4f}.'.format(np.mean(roc_aucs), np.std(roc_aucs)),
        '\nCV accuracy score   : {0:.4f}, std: {1:.4f}.'.format(np.mean(accuracies), np.std(accuracies)),
        '\nCV recall score     : {0:.4f}, std: {1:.4f}.'.format(np.mean(recalls), np.std(recalls)),
        '\nCV precision score  : {0:.4f}, std: {1:.4f}.'.format(np.mean(precisions), np.std(precisions)),
        '\nCV f1 score         : {0:.4f}, std: {1:.4f}.'.format(np.mean(f1_scores), np.std(f1_scores))
)

#ROC 
f = plt.figure(1)
plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'grey')
mean_tpr = np.mean(tprs, axis=0)
mean_auc_resampling_1 = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.4f)' % (np.mean(roc_aucs)),lw=2, alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LGB ROC curve by folds')
plt.legend(loc="lower right")

# PR plt
g = plt.figure(2)
plt.plot([0,1],[1,0],linestyle = '--',lw = 2,color = 'grey')
y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
precision, recall, _ = precision_recall_curve(y_real, y_proba)
plt.plot(recall, precision, color='blue',
         label=r'Mean P|R')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('P|R curve by folds')
plt.legend(loc="lower left")

# Confusion maxtrix & metrics
plt.rcParams["axes.grid"] = False
cm_resampling_1 = np.average(cms, axis=0)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cm_resampling_1, 
                      classes=class_names, 
                      title= 'LGB Confusion matrix [averaged/folds]')
plt.show()

plt.figure()
plot_confusion_matrix(cm_resampling_1, 
                      classes=class_names, 
                      title= 'LGB Confusion matrix [averaged/folds] percentage',percentage = True)
plt.show()


# # Resampling 2 :

# In[ ]:


plt.rcParams["axes.grid"] = True


oof_resampling_2 = np.zeros(len(df_train_resampling_2))
mean_fpr = np.linspace(0,1,100)
cms= []
tprs = []
aucs = []
y_real = []
y_proba = []
recalls = []
roc_aucs = []
f1_scores = []
accuracies = []
precisions_resampling_2 = []
predictions_resampling_2 = np.zeros(len(df_test))


i = 1
for train_idx, valid_idx in skf.split(df_train_resampling_2, df_train_resampling_2.isFraud.values):
    print("\nfold {}".format(i))
    trn_data = lgb.Dataset(df_train_resampling_2.iloc[train_idx][features].values,
                                   label=df_train_resampling_2.iloc[train_idx][target].values, feature_name=features 
                                   )
    val_data = lgb.Dataset(df_train_resampling_2.iloc[valid_idx][features].values,
                                   label=df_train_resampling_2.iloc[valid_idx][target].values,  feature_name=features
                                   )   
    
    clf = lgb.train(lgb_params, trn_data, num_boost_round = 500, valid_sets = [trn_data, val_data], verbose_eval = 250, early_stopping_rounds = 100)
    oof_resampling_2[valid_idx] = clf.predict(df_train_resampling_2.iloc[valid_idx][features].values) 
    
    predictions_resampling_2 += clf.predict(df_test[features]) / nfold
    
    # Scores 
    roc_aucs.append(roc_auc_score(df_train_resampling_2.iloc[valid_idx][target].values, oof_resampling_2[valid_idx]))
    accuracies.append(accuracy_score(df_train_resampling_2.iloc[valid_idx][target].values, oof_resampling_2[valid_idx].round()))
    recalls.append(recall_score(df_train_resampling_2.iloc[valid_idx][target].values, oof_resampling_2[valid_idx].round()))
    precisions.append(precision_score(df_train_resampling_2.iloc[valid_idx][target].values ,oof_resampling_2[valid_idx].round()))
    f1_scores.append(f1_score(df_train_resampling_2.iloc[valid_idx][target].values, oof_resampling_2[valid_idx].round()))
    
    # Roc curve by folds
    f = plt.figure(1)
    fpr, tpr, t = roc_curve(df_train_resampling_2.iloc[valid_idx][target].values, oof_resampling_2[valid_idx])
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (i,roc_auc))
    
    # Precion recall by folds
    g = plt.figure(2)
    precision, recall, _ = precision_recall_curve(df_train_resampling_2.iloc[valid_idx][target].values, oof_resampling_2[valid_idx])
    y_real.append(df_train_resampling_2.iloc[valid_idx][target].values)
    y_proba.append(oof_resampling_2[valid_idx])
    plt.plot(recall, precision, lw=2, alpha=0.3, label='P|R fold %d' % (i))  
    
    i= i+1
    
    # Confusion matrix by folds
    cms.append(confusion_matrix(df_train_resampling_2.iloc[valid_idx][target].values, oof_resampling_2[valid_idx].round()))
    


# Metrics
print(
        '\nCV roc score        : {0:.4f}, std: {1:.4f}.'.format(np.mean(roc_aucs), np.std(roc_aucs)),
        '\nCV accuracy score   : {0:.4f}, std: {1:.4f}.'.format(np.mean(accuracies), np.std(accuracies)),
        '\nCV recall score     : {0:.4f}, std: {1:.4f}.'.format(np.mean(recalls), np.std(recalls)),
        '\nCV precision score  : {0:.4f}, std: {1:.4f}.'.format(np.mean(precisions), np.std(precisions)),
        '\nCV f1 score         : {0:.4f}, std: {1:.4f}.'.format(np.mean(f1_scores), np.std(f1_scores))
)

#ROC 
f = plt.figure(1)
plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'grey')
mean_tpr = np.mean(tprs, axis=0)
mean_auc_resampling_2 = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.4f)' % (np.mean(roc_aucs)),lw=2, alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LGB ROC curve by folds')
plt.legend(loc="lower right")

# PR plt
g = plt.figure(2)
plt.plot([0,1],[1,0],linestyle = '--',lw = 2,color = 'grey')
y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
precision, recall, _ = precision_recall_curve(y_real, y_proba)
plt.plot(recall, precision, color='blue',
         label=r'Mean P|R')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('P|R curve by folds')
plt.legend(loc="lower left")

# Confusion maxtrix & metrics
plt.rcParams["axes.grid"] = False
cm_resampling_2 = np.average(cms, axis=0)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cm_resampling_2, 
                      classes=class_names, 
                      title= 'LGB Confusion matrix [averaged/folds]')
plt.show()

plt.figure()
plot_confusion_matrix(cm_resampling_2, 
                      classes=class_names, 
                      title= 'LGB Confusion matrix [averaged/folds] percentage',percentage = True)
plt.show()


# In[ ]:


mean_auc_ensembling = (0.6 * mean_auc) + 0.2 * (mean_auc_resampling_1 + mean_auc_resampling_2)
cm_ensembling = (0.6 * cm) + 0.2 * (cm_resampling_1 + cm_resampling_2)


# In[ ]:


print('Mean AUC Ensembling :', mean_auc_ensembling)

cm_ensembling = (0.6 * cm) + 0.2 * (cm_resampling_1 + cm_resampling_2)

plot_confusion_matrix(cm_ensembling, 
                      classes=class_names, 
                      title= 'LGB Confusion matrix [averaged/folds]')
plt.show()

plt.figure()
plot_confusion_matrix(cm_ensembling, 
                      classes=class_names, 
                      title= 'LGB Confusion matrix [averaged/folds] percentage',percentage = True)
plt.show()


# # Submissions :

# In[ ]:


sample_submission['isFraud'] = predictions
sample_submission.to_csv('submission_full_data.csv')


# In[ ]:


sample_submission['isFraud'] = predictions_resampling_1
sample_submission.to_csv('submission_resampling_1.csv')


# In[ ]:


sample_submission['isFraud'] = predictions_resampling_2
sample_submission.to_csv('submission_resampling_2.csv')


# In[ ]:


sample_submission['isFraud'] = (0.6 * predictions) + 0.2 * (predictions_resampling_1 + predictions_resampling_2)
sample_submission.to_csv('submission_ensembling.csv')

