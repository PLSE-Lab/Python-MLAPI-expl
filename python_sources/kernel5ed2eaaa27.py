# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import matplotlib.pyplot as plt
import gc
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_dataset = pd.read_csv("../input/dm-and-pr-ws1920-machine-learning-competition/train.csv")
test_dataset = pd.read_csv("../input/dm-and-pr-ws1920-machine-learning-competition/test.csv")

#delete the columns with too many null values
def get_too_many_null_attr(data):
        many_null_cols = [col for col in data.columns if data[col].isnull().sum() / data.shape[0] > 0.9]
        return many_null_cols
# don't have to delete any columns

#drop the target and id column
label = train_dataset['target']
train_dataset = train_dataset.drop(['target','id'],axis=1)

test_dataset = test_dataset.drop(['id'],axis=1)

#get all the amt column
AMT_columns = []
for i in range(1,7):
    AMT_columns.append("BILL_AMT{}".format(i))
    AMT_columns.append("PAY_AMT{}".format(i))

#get all the catergorical columns
categorical_columns = ['limit_balance','sex','education','marriage','n_children','age','profession','PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']

#transform data
def transform_data(data):
    for c in categorical_columns:
        for a in AMT_columns:
            data['{}_{}_std'.format(c,a)] = data['{}'.format(a)] / data.groupby(["{}".format(c)])['{}'.format(a)].transform('std')
    return data
train_dataset = transform_data(train_dataset)
test_dataset = transform_data(test_dataset) 

def log_AMT(data):
    for i in range(1,7):
        data["BILL_AMT{}_log".format(i)] = np.log(data["BILL_AMT{}".format(i)])
        data["PAY_AMT{}_log".format(i)] = np.log(data["PAY_AMT{}".format(i)])
    data = data.drop(AMT_columns,axis =1)
    return data

test_dataset = log_AMT(test_dataset)
train_dataset = log_AMT(train_dataset)

for each in train_dataset.columns:
        if train_dataset[each].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            train_dataset[each] = lbl.fit_transform(train_dataset[each].astype(str))
for each in test_dataset.columns:
        if test_dataset[each].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            test_dataset[each] = lbl.fit_transform(test_dataset[each].astype(str))


params = {'num_leaves': 423,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47,
         }

print("transform is finish,start training...")
NFOLDS = 5
folds = KFold(n_splits=NFOLDS,shuffle = False)
splits = folds.split(train_dataset, label)
score = 0
aucs = list()
# training_start_time = time()
for fold_n, (train_index, valid_index) in enumerate(splits):
    X_train, X_valid = train_dataset.iloc[train_index], train_dataset.iloc[valid_index]
    y_train, y_valid = label.iloc[train_index], label.iloc[valid_index]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    clf = lgb.train(params, dtrain, 5000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)
    aucs.append(clf.best_score['valid_1']['auc'])
   
#     print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
print('-' * 30)
print('Training has finished.')
print('Mean AUC:', np.mean(aucs))
print('-' * 30)
best_iter = clf.best_iteration
clf = lgb.LGBMClassifier(**params, num_boost_round=best_iter)
clf.fit(train_dataset, label)
sub = pd.read_csv('../input/dm-and-pr-ws1920-machine-learning-competition/sampleSubmission.csv')
sub['target'] = clf.predict_proba(test_dataset)[:, 1]
sub.to_csv('submission.csv', index=False)

#     print('Total training time is {}'


    
# merged_data['day_Amt_std'] = merged_data['Amt_log'] / merged_data.groupby(['day'])['Amt_log'].transform('std')


