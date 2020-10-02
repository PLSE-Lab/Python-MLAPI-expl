#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import category_encoders as ce

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm_notebook
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import lightgbm as lgb
import time
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn import metrics
import gc
import seaborn as sns
import warnings
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix, precision_score
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', 500)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in tqdm(df.columns):
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df


# In[ ]:


folder_path = '../input/'
train_identity = import_data(f'{folder_path}train_identity.csv')
train_transaction = import_data(f'{folder_path}train_transaction.csv')
test_identity = import_data(f'{folder_path}test_identity.csv')
test_transaction = import_data(f'{folder_path}test_transaction.csv')
sub = import_data(f'{folder_path}sample_submission.csv')
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')


# In[ ]:


def device_proc(x):
    if x == 'Windows':
        return 'win'
    elif x == 'iOS Device' or x == 'MacOS':
        return 'Mac'
    else:
        return 'Other'
    
def browser_proc(x):
    try:
        x = x.split(' ')
        return x[0]
    except: 
        return x

def os_proc(x):
    try:
        x = x.split(' ')
        return x[0]
    except:
        return x
    
def email1_proc(x):
    try:
        x = x.split('.')
        return x[0]
    except:
        return x
    
def data_proc(train):
    train['DeviceInfo'] = train.DeviceInfo.apply(lambda x: device_proc(x))
    train['id_31'] = train.id_31.apply(lambda x: browser_proc(x))
    train['id_30'] = train.id_30.apply(lambda x: os_proc(x))
    train['P_emaildomain'] = train.P_emaildomain.apply(lambda x: email1_proc(x))
    train['R_emaildomain'] = train.R_emaildomain.apply(lambda x: email1_proc(x))

    return train
train = data_proc(train)
test = data_proc(test)


# In[ ]:


y = train.isFraud.values
train.drop(['TransactionDT', 'TransactionID', 'isFraud'], axis=1, inplace=True)
test.drop(['TransactionDT', 'TransactionID'], axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


usecol_feature_permutation = ['C1', 'C13', 'card1', 'card2', 'TransactionAmt', 'C14', 'card6', 'addr1', 'D2', 'D15', 'C11', 'V91', 'V257', 'D1', 'P_emaildomain', 'R_emaildomain', 'card5', 'ProductCD', 'D4', 'C6', 'C2', 'D10', 'M4', 'V294', 'V70', 'V317', 'V283', 'V310', 'card3', 'id_17', 'id_31', 'dist1', 'V48', 'id_33', 'D11', 'M6', 'id_20', 'D8', 'id_19', 'C9', 'C12', 'M5', 'C5', 'id_01', 'C8', 'V281', 'V313', 'card4', 'V308', 'id_13', 'M3', 'V55', 'id_02', 'V288', 'V189', 'V62', 'V280', 'V307', 'C10', 'V223', 'V312', 'id_38', 'V61', 'V314', 'M9', 'V131', 'id_05', 'V87', 'V296', 'V37', 'V130', 'id_06', 'V165', 'V66', 'id_30', 'V76', 'M2', 'V282', 'V322', 'dist2', 'C4', 'V75', 'V67', 'V97', 'V82', 'D9', 'D13', 'D6', 'V187', 'V90', 'V77', 'id_14', 'V277', 'V49', 'V13', 'V78', 'V139', 'V53', 'V128', 'DeviceType', 'V151', 'V5', 'id_37', 'V44', 'V285', 'V166', 'V306', 'addr2', 'id_03', 'V287', 'V266', 'V86', 'V56', 'V38', 'C7', 'V17', 'V264', 'V81', 'M8', 'V207', 'V204', 'V140', 'V99']


# In[ ]:


encoder = ce.WOEEncoder(verbose=1,  drop_invariant=False, return_df=True, 
                        handle_unknown='value', handle_missing='value', random_state=42, 
                        randomized=False, sigma=0.05, regularization=1.0)
encoder.fit(train, y)
train = encoder.transform(train)
test = encoder.transform(test)


# In[ ]:


train = train.fillna(0)
test = test.fillna(0)


# In[ ]:


target = y
gc.collect()


# In[ ]:


params = {'colsample_bytree': 0.10077042112439025, 'max_depth': 95, 'num_leaves': 481, 'reg_alpha': 0.0038938292050716425, 'reg_lambda': 4.961057796456089, 'bagging_fraction': 0.6057333586649449, 'feature_fraction': 0.6004878444394529, 'min_data_in_leaf': 81, 'min_sum_hessian_in_leaf': 0.11844994003313261, 'min_gain_to_split': 0.5222971942325503, 'feature_fraction_seed': 42, 'bagging_seed': 42, 'drop_seed': 42, 'data_random_seed': 42, 'boosting_type': 'gbdt', 'objective': 'binary', 'learning_rate': 0.2, 'seed': 42, 'save_binary': True, 'boost_from_average': True}
import eli5
from IPython.display import display
from eli5.permutation_importance import get_score_importances
from eli5.sklearn import PermutationImportance

fold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
for fold_, (trn_idx, val_idx) in enumerate(fold.split(train, target)):
    print("%%%%%%%%%%%%%%Fold idx:{}%%%%%%%%%%%%%%%".format(fold_ + 1))

    trn, trg = train.iloc[trn_idx], target[trn_idx]
    trn_data = lgb.Dataset(trn, label = trg)
    val_data = lgb.Dataset(train.iloc[val_idx], label = target[val_idx])
    clf = lgb.LGBMClassifier(**params, n_estimators = 1000, silent=-1, verbose=-1)
    clf.fit(train.iloc[trn_idx], target[trn_idx], verbose=100)    
    print('Trained')
    perm = PermutationImportance(clf,scoring=None, n_iter=1, random_state=42, cv=None, refit=False).fit(train.iloc[val_idx], target[val_idx])
    tmp = eli5.show_weights(perm)
    display(eli5.show_weights(perm, top = len(list(train.columns)), feature_names = list(train.columns)))


# In[ ]:


train = train[usecol_feature_permutation]
test = test[usecol_feature_permutation]


# In[ ]:


n_fold = 5
fold = KFold(n_splits=n_fold, random_state=42, shuffle=True)
params = {'colsample_bytree': 0.10077042112439025, 'max_depth': 95, 'num_leaves': 481, 'reg_alpha': 0.0038938292050716425, 'reg_lambda': 4.961057796456089, 'bagging_fraction': 0.6057333586649449, 'feature_fraction': 0.6004878444394529, 'min_data_in_leaf': 81, 'min_sum_hessian_in_leaf': 0.11844994003313261, 'min_gain_to_split': 0.5222971942325503, 'feature_fraction_seed': 42, 'bagging_seed': 42, 'drop_seed': 42, 'data_random_seed': 42, 'boosting_type': 'gbdt', 'objective': 'binary', 'learning_rate': 0.2, 'seed': 42, 'save_binary': True, 'boost_from_average': True}
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
for fold_, (trn_idx, val_idx) in enumerate(fold.split(train, y)):
    print("Fold idx:{}".format(fold_ + 1))
    trn, trg = train.iloc[trn_idx], y[trn_idx]
    trn_data = lgb.Dataset(trn, label = trg)
    val_enc = train.iloc[val_idx]
    val_data = lgb.Dataset(val_enc, label = y[val_idx])
    clf = lgb.train(params, trn_data, 5_000, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 500)
    oof[val_idx] = clf.predict(val_enc, num_iteration=clf.best_iteration)
    predictions += clf.predict(test,  num_iteration=clf.best_iteration) / fold.n_splits #,#scaler.transform(encoder.transform(test))
    gc.collect()
print("CV score: {:.6f}".format(roc_auc_score(y, oof)))


# In[ ]:


import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


cnf_matrix = confusion_matrix(y,np.round(oof))
plot_confusion_matrix(cnf_matrix, classes=["0", "1"],
                      title='Confusion matrix, lgbm1')


# In[ ]:


sub['isFraud'] = predictions
sub.to_csv('submission.csv', index=False)

sub.isFraud.hist(bins=10)


# In[ ]:




