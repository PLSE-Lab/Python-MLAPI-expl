#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('wget train.zip https://datahack-prod.s3.amazonaws.com/train_file/train_aox2Jxw.zip')


# In[ ]:


get_ipython().system('wget test.csv https://datahack-prod.s3.amazonaws.com/test_file/test_bqCt9Pv.csv')


# In[ ]:


get_ipython().system('wget sample.csv https://datahack-prod.s3.amazonaws.com/sample_submission/sample_submission_24jSKY6.csv')


# In[ ]:


get_ipython().system('unzip train_aox2Jxw.zip')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test_bqCt9Pv.csv')
sample = pd.read_csv('sample_submission_24jSKY6.csv')
data_dict = pd.read_excel('Data Dictionary.xlsx')


# In[ ]:


data_dict


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape, test.shape


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train['Employment.Type'].value_counts()


# In[ ]:


train['Employment.Type'].replace(np.nan, 'Not Provided', inplace=True)
test['Employment.Type'].replace(np.nan, 'Not Provided', inplace=True)


# In[ ]:


train.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "train['Date.of.Birth'] = pd.to_datetime(train['Date.of.Birth'])\ntrain['DisbursalDate'] = pd.to_datetime(train['DisbursalDate'])\ntrain['age'] = (train['DisbursalDate'] - train['Date.of.Birth']).dt.days\ntrain['age'] /= 365\n\ntest['Date.of.Birth'] = pd.to_datetime(test['Date.of.Birth'])\ntest['DisbursalDate'] = pd.to_datetime(test['DisbursalDate'])\ntest['age'] = (test['DisbursalDate'] - test['Date.of.Birth']).dt.days\ntest['age'] /= 365")


# In[ ]:


s = '10yrs 10mon'
s[0:s.find('yrs')], s[s.find(' ')+1:s.find('mon')]


# In[ ]:


get_ipython().run_cell_magic('time', '', "train['AVERAGE.ACCT.AGE'] = train['AVERAGE.ACCT.AGE'].apply(lambda s: int(s[0 : s.find('yrs')]) + int(s[s.find(' ')+1:s.find('mon')])/12)\ntest['AVERAGE.ACCT.AGE'] = test['AVERAGE.ACCT.AGE'].apply(lambda s: int(s[0 : s.find('yrs')]) + int(s[s.find(' ')+1:s.find('mon')])/12)")


# In[ ]:


test.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "train['CREDIT.HISTORY.LENGTH'] = train['CREDIT.HISTORY.LENGTH'].apply(lambda s: int(s[0 : s.find('yrs')]) + int(s[s.find(' ')+1:s.find('mon')])/12)\ntest['CREDIT.HISTORY.LENGTH'] = test['CREDIT.HISTORY.LENGTH'].apply(lambda s: int(s[0 : s.find('yrs')]) + int(s[s.find(' ')+1:s.find('mon')])/12)")


# In[ ]:


train.head()


# In[ ]:


uid = test['UniqueID']
cols = ['UniqueID', 'Date.of.Birth', 'DisbursalDate']
train.drop(cols, axis=1, inplace=True)
test.drop(cols, axis=1, inplace=True)


# In[ ]:


fig, axs = plt.subplots(1,2, figsize = (15,5))
train['Employment.Type'].value_counts().plot.bar(ax = axs[0])
test['Employment.Type'].value_counts().plot.bar(ax = axs[1])


# In[ ]:


fig, axs = plt.subplots(1,2, figsize = (15,5))
train['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts().plot.bar(ax = axs[0])
test['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts().plot.bar(ax = axs[1])


# In[ ]:


train['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts()


# In[ ]:


test['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts()


# In[ ]:


train.shape


# In[ ]:


train = train[(train['PERFORM_CNS.SCORE.DESCRIPTION'] != 'Not Scored: More than 50 active Accounts found')]


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


# from sklearn.preprocessing import LabelEncoder
# cat = ['PERFORM_CNS.SCORE.DESCRIPTION', 'Employment.Type']
# for i in cat:
#     lb = LabelEncoder()
#     lb.fit(train[i])
#     train[i] = lb.transform(train[i])
#     test[i] = lb.transform(test[i])    


# In[ ]:


cat = ['PERFORM_CNS.SCORE.DESCRIPTION', 'Employment.Type']
for i in cat:
    dummy = pd.get_dummies(train[i])
    train = pd.concat([train, dummy], axis = 1)
    train.drop(i, axis = 1, inplace = True)
    
    dummy = pd.get_dummies(test[i])
    test = pd.concat([test, dummy], axis = 1)
    test.drop(i, axis = 1, inplace = True)


# In[ ]:


train.corr()


# In[ ]:


corr_matrix = train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
print(to_drop)


# In[ ]:


train.drop(to_drop, axis=1, inplace=True)
test.drop(to_drop, axis=1, inplace=True)


# In[ ]:


train['loan_default'].value_counts()


# In[ ]:


train.head()


# In[ ]:


cols = train.columns


# In[ ]:


from sklearn.utils import resample
tr1 = train[train['loan_default'] == 0]
tr2 = train[train['loan_default'] != 0]
print(tr1.shape,tr2.shape,train.shape)
tr1 = resample(tr1, replace = False, n_samples = 130000, random_state = 51)
train_downsample = pd.concat([tr1, tr2])


# In[ ]:


from sklearn.utils import resample
tr1 = train[train['loan_default'] == 1]
tr2 = train[train['loan_default'] != 1]
print(tr1.shape,tr2.shape,train.shape)
tr1 = resample(tr1, replace = True, n_samples = 100000, random_state = 51)
train_upsample = pd.concat([tr1, tr2])


# In[ ]:


train_downsample['loan_default'].value_counts()


# In[ ]:


train_upsample['loan_default'].value_counts()


# In[ ]:


y = train['loan_default']
train.drop(['loan_default'], axis=1, inplace=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'cols = train.columns\nfrom imblearn.over_sampling import SMOTE\nsm = SMOTE(random_state=51)\ntrain_smote, y_smote = sm.fit_resample(train, y)')


# In[ ]:


corr_matrix = train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
print(to_drop)

train.drop(to_drop, axis=1, inplace=True)
test.drop(to_drop, axis=1, inplace=True)


# In[ ]:


corr_matrix = train_smote.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
print(to_drop)

train_smote.drop(to_drop, axis=1, inplace=True)
# test_smote.drop(to_drop, axis=1, inplace=True)


# In[ ]:


corr_matrix = train_upsample.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
print(to_drop)

train_upsample.drop(to_drop, axis=1, inplace=True)
# test_upsample.drop(to_drop, axis=1, inplace=True)


# In[ ]:


corr_matrix = train_downsample.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
print(to_drop)

train_downsample.drop(to_drop, axis=1, inplace=True)
# test_downsample.drop(to_drop, axis=1, inplace=True)


# In[ ]:


y_downsample = train_downsample['loan_default']
train_downsample.drop(['loan_default'], axis=1, inplace=True)

y_upsample = train_upsample['loan_default']
train_upsample.drop(['loan_default'], axis=1, inplace=True)


# In[ ]:


sample.head()


# In[ ]:


from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
print(class_weights)
class_weight_dict = dict(enumerate(class_weights))
print(class_weight_dict)


# In[ ]:





# In[ ]:


# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# scaler = MinMaxScaler()
# train = scaler.fit_transform(train)
# test = scaler.transform(test)

# scaler = MinMaxScaler()
# train_smote = scaler.fit_transform(train_smote)
# test_smote = scaler.transform(test_smote)

# scaler = MinMaxScaler()
# train_downsample = scaler.fit_transform(train_downsample)
# test_downsample = scaler.transform(test_downsample)

# scaler = MinMaxScaler()
# train_upsample = scaler.fit_transform(train_upsample)
# test_upsample = scaler.transform(test_upsample)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from lightgbm import LGBMClassifier\nclf1 = LGBMClassifier(random_state=25)\nclf1.fit(train, y)\n\n# clf2 = LGBMClassifier(random_state=25, class_weight={0: 0.6386298893393229, 1: 1.5})\n# clf2.fit(train, y)\n\n# clf3 = LGBMClassifier(random_state=25)\n# clf3.fit(train_smote, y_smote)\n\n# clf4 = LGBMClassifier( random_state=25)\n# clf4.fit(train_downsample, y_downsample)\n\n# clf5 = LGBMClassifier( random_state=25)\n# clf5.fit(train_upsample, y_upsample)')


# In[ ]:


yp1 = clf1.predict_proba(test)[:, 1]
# yp2 = clf2.predict_proba(test)[:, 1]
# yp3 = clf3.predict_proba(test)[:, 1]
# yp4 = clf4.predict_proba(test)[:, 1]
# yp5 = clf5.predict_proba(test)[:, 1]

# yp = (yp1 + yp2 + yp3 + yp4 + yp5)/5

sub = pd.DataFrame({'UniqueID' : uid, 'loan_default' : yp1})
sub.to_csv('LGB.csv', index = False)
cnt = 0
for i in yp1:
    if(i >= 0.5):
        cnt += 1
print(cnt)


# In[ ]:


import seaborn as sns
feature_imp = pd.DataFrame(sorted(zip(clf1.feature_importances_, cols)), columns=['Value','Feature'])
plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()


# In[ ]:


# %%time
# from xgboost import XGBClassifier
# clf = XGBClassifier(n_estimators = 500)
# clf.fit(train, y)


# In[ ]:


# # yp = clf.predict_proba(test)[:, 1]
# # sub = pd.DataFrame({'UniqueID' : uid, 'loan_default' : yp})
# # sub.to_csv('XGB.csv', index = False)
# cnt = 0
# for i in yp:
#     if(i >= 0.5):
#         cnt += 1
# print(cnt)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import RandomizedSearchCV, GridSearchCV\nfrom sklearn.metrics import roc_auc_score\nfrom sklearn.model_selection import StratifiedKFold\n\nparams = {\n    \'n_estimators\': [100, 250, 400, 75],\n    \'max_depth\': [7, 9, 11, 13],\n    \'learning_rate\': [0.1, 0.01, 0.05],\n#     \'num_leaves\': [4000, 8000]\n#     \'min_child_weight\': [1, 3, 5],\n#     \'subsample\': [0.6, 0.8, 1.0],\n#     \'colsample_bytree\': [0.6, 0.8, 1.0],\n}\n\nclf1 = LGBMClassifier(random_state=51, class_weight={0: 0.6386298893393229, 1: 1.5})\n# clf1.fit(train, y)\n\nclf2 = LGBMClassifier(random_state=51)\n# clf2.fit(train, y)\n\nclf3 = LGBMClassifier(random_state=51)\n# clf3.fit(train_smote, y_smote)\n\nclf4 = LGBMClassifier( random_state=51)\n# clf4.fit(train_downsample, y_downsample)\n\nclf5 = LGBMClassifier( random_state=51)\n# clf5.fit(train_upsample, y_upsample)\n\n# clf = LGBMClassifier(random_state=25)\nskf = StratifiedKFold(n_splits=3, shuffle = True, random_state = 51)\nprint("--------------------       clf1          ---------------------------")\ngrid1 = GridSearchCV(estimator=clf1, param_grid=params, scoring=\'roc_auc\', cv=skf.split(train, y), verbose=3 )\ngrid1.fit(train, y)\nprint(\'\\n All results:\')\nprint(grid1.cv_results_)\nprint(\'\\n Best estimator:\')\nprint(grid1.best_estimator_)\nprint(\'\\n Best score:\')\nprint(grid1.best_score_ * 2 - 1)\nprint(\'\\n Best parameters:\')\nprint(grid1.best_params_)\n\nprint("--------------------       clf2          ---------------------------")\ngrid2 = GridSearchCV(estimator=clf2, param_grid=params, scoring=\'roc_auc\', cv=skf.split(train, y), verbose=3 )\ngrid2.fit(train, y)\nprint(\'\\n All results:\')\nprint(grid2.cv_results_)\nprint(\'\\n Best estimator:\')\nprint(grid2.best_estimator_)\nprint(\'\\n Best score:\')\nprint(grid2.best_score_ * 2 - 1)\nprint(\'\\n Best parameters:\')\nprint(grid2.best_params_)\n\nprint("--------------------       clf3          ---------------------------")\ngrid3 = GridSearchCV(estimator=clf3, param_grid=params, scoring=\'roc_auc\', cv=skf.split(train_smote, y_smote), verbose=3 )\ngrid3.fit(train_smote, y_smote)\nprint(\'\\n All results:\')\nprint(grid3.cv_results_)\nprint(\'\\n Best estimator:\')\nprint(grid3.best_estimator_)\nprint(\'\\n Best score:\')\nprint(grid3.best_score_ * 2 - 1)\nprint(\'\\n Best parameters:\')\nprint(grid3.best_params_)\n\nprint("--------------------       clf5          ---------------------------")\ngrid5 = GridSearchCV(estimator=clf5, param_grid=params, scoring=\'roc_auc\', cv=skf.split(train_upsample, y_upsample), verbose=3 )\ngrid5.fit(train_upsample, y_upsample)\nprint(\'\\n All results:\')\nprint(grid5.cv_results_)\nprint(\'\\n Best estimator:\')\nprint(grid5.best_estimator_)\nprint(\'\\n Best score:\')\nprint(grid5.best_score_ * 2 - 1)\nprint(\'\\n Best parameters:\')\nprint(grid5.best_params_)\n\n\nprint("--------------------       clf4          ---------------------------")\ngrid4 = GridSearchCV(estimator=clf4, param_grid=params, scoring=\'roc_auc\', cv=skf.split(train_downsample, y_downsample), verbose=3 )\ngrid4.fit(train_downsample, y_downsample)\nprint(\'\\n All results:\')\nprint(grid4.cv_results_)\nprint(\'\\n Best estimator:\')\nprint(grid4.best_estimator_)\nprint(\'\\n Best score:\')\nprint(grid4.best_score_ * 2 - 1)\nprint(\'\\n Best parameters:\')\nprint(grid4.best_params_)')


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n\n\n\n\nyp1 = grid1.best_estimator_.predict_proba(test)[:, 1]\nyp2 = grid2.best_estimator_.predict_proba(test)[:, 1]\nyp3 = grid3.best_estimator_.predict_proba(test)[:, 1]\nyp4 = grid4.best_estimator_.predict_proba(test)[:, 1]\nyp5 = grid5.best_estimator_.predict_proba(test)[:, 1]\n\n\nyp = (yp1 + yp2 + yp3 + yp4 + yp5)/5\n\nsub = pd.DataFrame({'UniqueID' : uid, 'loan_default' : yp})\nsub.to_csv('LGB_grid1.csv', index = False)")


# In[ ]:


cnt = 0
for i in yp:
    if(i >= 0.5):
        cnt += 1
print(cnt)

