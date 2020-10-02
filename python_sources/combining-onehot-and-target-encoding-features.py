#!/usr/bin/env python
# coding: utf-8

# This Notebooke is superposition of these two published Notebooks
# 
# * https://www.kaggle.com/peterhurford/why-not-logistic-regression
# * https://www.kaggle.com/merckel/target-encoding-and-lightgbm
# 
# I have done some model tunning and processing on the top of that.

# In[ ]:





# In[ ]:


get_ipython().run_cell_magic('time', '', "\nimport pandas as pd\nimport numpy as np\nimport category_encoders as ce\n\n# Load data\ntrain = pd.read_csv('../input/cat-in-the-dat/train.csv')\ntest = pd.read_csv('../input/cat-in-the-dat/test.csv')\n\nprint(train.shape)\nprint(test.shape)")


# In[ ]:


train.head()


# In[ ]:


from sklearn.model_selection import StratifiedKFold


# In[ ]:


# from pandas.api.types import CategoricalDtype 

# ord_1 = CategoricalDtype(categories=['Novice', 'Contributor','Expert', 
#                                      'Master', 'Grandmaster'], ordered=True)
# ord_2 = CategoricalDtype(categories=['Freezing', 'Cold', 'Warm', 'Hot',
#                                      'Boiling Hot', 'Lava Hot'], ordered=True)
# ord_3 = CategoricalDtype(categories=['a', 'b', 'c', 'd', 'e', 'f', 'g',
#                                      'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'], ordered=True)
# ord_4 = CategoricalDtype(categories=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
#                                      'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
#                                      'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], ordered=True)

# train[["ord_1", "ord_2", "ord_3", "ord_4"]] = train[["ord_1", "ord_2", "ord_3", "ord_4"]].astype("category")
# test[["ord_1", "ord_2", "ord_3", "ord_4"]] = test[["ord_1", "ord_2", "ord_3", "ord_4"]].astype("category")
# train["ord_1"] = train.ord_1.astype(ord_1)
# train["ord_2"] = train.ord_2.astype(ord_2)
# train["ord_3"] = train.ord_3.astype(ord_3)
# train["ord_4"] = train.ord_4.astype(ord_4)

# test["ord_1"] = test.ord_1.cat.codes
# test["ord_2"] = test.ord_2.cat.codes
# test["ord_3"] = test.ord_3.cat.codes
# test["ord_4"] = test.ord_4.cat.codes


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Subset\ntarget = train['target']\ntrain_id = train['id']\ntest_id = test['id']\ntrain.drop(['target', 'id'], axis=1, inplace=True)\ntest.drop('id', axis=1, inplace=True)\n\nprint(train.shape)\nprint(test.shape)")


# In[ ]:


test.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# One Hot Encode\ntraintest = pd.concat([train, test])\ndummies = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True)\ntrain_ohe = dummies.iloc[:train.shape[0], :]\ntest_ohe = dummies.iloc[train.shape[0]:, :]\n\nprint(train_ohe.shape)\nprint(test_ohe.shape)')


# In[ ]:


train_ohe.head(2)


# **Target encoding for cateogrocal Features**
# 

# In[ ]:


cat_feat_to_encode = train.columns.tolist()
smoothing=50.0

oof = pd.DataFrame([])
for tr_idx, oof_idx in StratifiedKFold(
    n_splits=5, random_state=1, shuffle=True).split(
        train, target):
    ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)
    ce_target_encoder.fit(train.iloc[tr_idx, :], target.iloc[tr_idx])
    oof = oof.append(ce_target_encoder.transform(train.iloc[oof_idx, :]), ignore_index=False)

ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)
ce_target_encoder.fit(train, target)
train_enc = oof.sort_index() 
test_enc = ce_target_encoder.transform(test)


# In[ ]:


train_enc.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# To be honest, I am a bit confused what is going on with the new sparse dataframe interface in Pandas v0.25\n\n# It looks like `sparse = True` in `get_dummies` no longer makes anything sparse, and we have to explicitly convert\n# like this...\n\n# If you don't do this, the model takes forever... it is much much faster on sparse data!\n\ntrain_ohe = train_ohe.sparse.to_coo().tocsr()\ntest_ohe = test_ohe.sparse.to_coo().tocsr()")


# In[ ]:


import scipy
scipy.sparse.hstack


# **Merge onehot encoded features with target encoded features**

# In[ ]:


train_merge = scipy.sparse.hstack([train_ohe, train_enc.values]).tocsr()
test_merge = scipy.sparse.hstack([test_ohe, test_enc.values]).tocsr()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nfrom sklearn.model_selection import KFold\nfrom sklearn.metrics import roc_auc_score as auc\nfrom sklearn.linear_model import LogisticRegression\n\n# Model\ndef run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label=\'model\'):\n    kf = KFold(n_splits=10)\n    fold_splits = kf.split(train, target)\n    cv_scores = []\n    pred_full_test = 0\n    pred_train = np.zeros((train.shape[0]))\n    i = 1\n    for dev_index, val_index in fold_splits:\n        print(\'Started \' + label + \' fold \' + str(i) + \'/10\')\n        #print("dev_index", dev_index)\n        dev_X, val_X = train[dev_index], train[val_index]\n        dev_y, val_y = target[dev_index], target[val_index]\n        params2 = params.copy()\n        pred_val_y, pred_test_y = model_fn(dev_X, dev_y, val_X, val_y, test, params2)\n        pred_full_test = pred_full_test + pred_test_y\n        pred_train[val_index] = pred_val_y\n        if eval_fn is not None:\n            cv_score = eval_fn(val_y, pred_val_y)\n            cv_scores.append(cv_score)\n            print(label + \' cv score {}: {}\'.format(i, cv_score))\n        i += 1\n    print(\'{} cv scores : {}\'.format(label, cv_scores))\n    print(\'{} cv mean score : {}\'.format(label, np.mean(cv_scores)))\n    print(\'{} cv std score : {}\'.format(label, np.std(cv_scores)))\n    pred_full_test = pred_full_test / 5.0\n    results = {\'label\': label,\n              \'train\': pred_train, \'test\': pred_full_test,\n              \'cv\': cv_scores}\n    return results\n\n\ndef runLR(train_X, train_y, test_X, test_y, test_X2, params):\n    print(\'Train LR\')\n    model = LogisticRegression(**params)\n    model.fit(train_X, train_y)\n    print(\'Predict 1/2\')\n    pred_test_y = model.predict_proba(test_X)[:, 1]\n    print(\'Predict 2/2\')\n    pred_test_y2 = model.predict_proba(test_X2)[:, 1]\n    return pred_test_y, pred_test_y2\n\nfrom catboost import CatBoostClassifier\nimport lightgbm as lgb\ndef runCat(train_X, train_y, test_X, test_y, test_X2, params):\n    print(\'Train LR\')\n    model = CatBoostClassifier(**params)\n    model.fit(train_X, train_y)\n    print(\'Predict 1/2\')\n    pred_test_y = model.predict_proba(test_X)[:, 1]\n    print(\'Predict 2/2\')\n    pred_test_y2 = model.predict_proba(test_X2)[:, 1]\n    return pred_test_y, pred_test_y2\n\ndef runLgb(train_X, train_y, test_X, test_y, test_X2, params):\n    model = lgb.train(\n    params={\n        \'max_depth\': 3, \n        \'num_leaves\': 150,\n        \'reg_alpha\': 0.6, \n        \'reg_lambda\': 0.6,\n        \'objective\': \'binary\',\n        "boosting_type": "gbdt",\n        "metric": \'auc\',\n        "verbosity": -1,\n        \'random_state\': 1,\n        \'lr\': 0.01\n    },\n    train_set=lgb.Dataset(train_X, label=train_y),\n    num_boost_round=700)\n    \n    print(\'Predict 1/2\')\n    pred_test_y = model.predict(test_X)\n    print(\'Predict 2/2\')\n    pred_test_y2 = model.predict(test_X2)\n    return pred_test_y, pred_test_y2\n\n\nmodel = CatBoostClassifier(learning_rate=0.006, iterations=1000, thread_count=32,\n                           eval_metric=\'Accuracy\')\n\nlr_params = {\'solver\': \'lbfgs\', \'C\': 0.1, "max_iter":500, \'thread_count\':32, "eval_metric":"AUC"}\ncat_params = {"iterations":500, \'learning_rate\': 0.006, \'thread_count\':32, \'eval_metric\':\'AUC\'}\n#results = run_cv_model(train_ohe, test_ohe, target, runLR, lr_params, auc, \'lr\')\n#results = run_cv_model(train_merge, test_merge, target, runLR, lr_params, auc, \'lr\')\n#results = run_cv_model(train_merge.toarray(), test_merge.toarray(), target, runCat, cat_params, auc, \'catboost\')\nresults = run_cv_model(train_merge, test_merge, target, runLgb, {}, auc, \'lgb\')')


# In[ ]:


# Make submission
submission = pd.DataFrame({'id': test_id, 'target': results['test']})
submission.to_csv('submission.csv', index=False)
