#!/usr/bin/env python
# coding: utf-8

# Catboost is known as a convenient and effective tool for handling categorical features. Let's build a Catboost baseline by specifying the categorical feature indices to the model. I have also tried some cyclical and ordinal encoding methods. 
# 
# This is my first try on Catboost. If you have found any mistake, please write your comments below. Thank you very much.
# 
# This notebook is modified from [Why Not Logistic Regression?][1]. Please upvote that notebook first.
# 
# [1]: https://www.kaggle.com/peterhurford/why-not-logistic-regression

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nimport pandas as pd\nimport numpy as np\nimport string\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nsns.set()\n%matplotlib inline\n\n# Load data\ntrain = pd.read_csv('../input/cat-in-the-dat/train.csv')\ntest = pd.read_csv('../input/cat-in-the-dat/test.csv')\n\nprint(train.shape)\nprint(test.shape)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Subset\ntarget = train['target']\ntrain_id = train['id']\ntest_id = test['id']\ntrain.drop(['target', 'id'], axis=1, inplace=True)\ntest.drop('id', axis=1, inplace=True)\n\nprint(train.shape)\nprint(test.shape)")


# Transfer the cyclical features into two dimensional sin-cos features

# In[ ]:


# # Transfer the cyclical features into two dimensional sin-cos features
# # https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning
# def cyclical_encode(data, col, max_val):
#     data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
#     data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
#     return data

# train = cyclical_encode(train, 'day', 7)
# test = cyclical_encode(test, 'day', 7) 

# train = cyclical_encode(train, 'month', 12)
# test = cyclical_encode(test, 'month', 12)

# train.drop(['day', 'month'], axis=1, inplace=True)
# test.drop(['day', 'month'], axis=1, inplace=True)


# Just do ordinal encoding for 'ord_1' to 'ord_4' by order. 'ord_0' contains numerical values so we do not need to encode it again.

# In[ ]:


# First, I encode ord_1 to ord_4 since the numbers of their unique values are small 
mapper_ord_1 = {'Novice': 1, 
                'Contributor': 2,
                'Expert': 3, 
                'Master': 4, 
                'Grandmaster': 5}

# https://www.kaggle.com/asimandia/let-s-try-some-feature-engineering
traintest = pd.concat([train, test])
train['ord_1_count'] = train['ord_1'].map(traintest['ord_1'].value_counts().to_dict())
test['ord_1_count'] = test['ord_1'].map(traintest['ord_1'].value_counts().to_dict())

mapper_ord_2 = {'Freezing': 1, 
                'Cold': 2, 
                'Warm': 3, 
                'Hot': 4,
                'Boiling Hot': 5, 
                'Lava Hot': 6}

mapper_ord_3 = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 
                'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15}

mapper_ord_4 = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 
                'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15,
                'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 
                'W': 23, 'X': 24, 'Y': 25, 'Z': 26}

for col, mapper in zip(['ord_1', 'ord_2', 'ord_3', 'ord_4'], [mapper_ord_1, mapper_ord_2, mapper_ord_3, mapper_ord_4]):
    train[col+'_oe'] = train[col].replace(mapper)
    test[col+'_oe'] = test[col].replace(mapper)
    train.drop(col, axis=1, inplace=True)
    test.drop(col, axis=1, inplace=True)


# Here I have also tried to encode 'ord_5'.
# 
# **Option 1**: Add up the indices of two letters in string.ascii_letters
# 
# **Option 2**: Join the indices of two letters in string.ascii_letters
# 
# **Option 3**: Split 'ord_5' into two new columns using the indices of two letters in string.ascii_letters, separately
# 
# **Option 4**: Simply sort their values by string
# 

# In[ ]:


# # Then encode 'ord_5' using ACSII values

# # Option 1: Add up the indices of two letters in string.ascii_letters
# train['ord_5_oe_add'] = train['ord_5'].apply(lambda x:sum([(string.ascii_letters.find(letter)+1) for letter in x]))
# test['ord_5_oe_add'] = test['ord_5'].apply(lambda x:sum([(string.ascii_letters.find(letter)+1) for letter in x]))

# # Option 2: Join the indices of two letters in string.ascii_letters
# train['ord_5_oe_join'] = train['ord_5'].apply(lambda x:float(''.join(str(string.ascii_letters.find(letter)+1) for letter in x)))
# test['ord_5_oe_join'] = test['ord_5'].apply(lambda x:float(''.join(str(string.ascii_letters.find(letter)+1) for letter in x)))

# # Option 3: Split 'ord_5' into two new columns using the indices of two letters in string.ascii_letters, separately
train['ord_5_oe1'] = train['ord_5'].apply(lambda x:(string.ascii_letters.find(x[0])+1))
test['ord_5_oe1'] = test['ord_5'].apply(lambda x:(string.ascii_letters.find(x[0])+1))

train['ord_5_oe2'] = train['ord_5'].apply(lambda x:(string.ascii_letters.find(x[1])+1))
test['ord_5_oe2'] = test['ord_5'].apply(lambda x:(string.ascii_letters.find(x[1])+1))

# Option 4: Simply sort their values by string
# https://www.kaggle.com/c/cat-in-the-dat/discussion/105702#latest-607652
ord_5 = sorted(list(set(train['ord_5'].values)))
ord_5 = dict(zip(ord_5, range(len(ord_5))))
train.loc[:, 'ord_5'] = train['ord_5'].apply(lambda x: ord_5[x]).astype(float)
test.loc[:, 'ord_5'] = test['ord_5'].apply(lambda x: ord_5[x]).astype(float)

# train.drop('ord_5', axis=1, inplace=True)
# test.drop('ord_5', axis=1, inplace=True)


# In[ ]:


# Transfer the dtypes of encoded ordinal features into float64
for col in ['ord_0', 'ord_1_oe', 'ord_2_oe', 'ord_3_oe', 'ord_4_oe', 'ord_5_oe1', 'ord_5_oe2', 'ord_1_count']: #, 'ord_5_oe_add', 'ord_5_oe_join'
    train[col]= train[col].astype('float64')
    test[col]= test[col].astype('float64')


# In[ ]:


# # Do some ordinal encoding
# from sklearn.preprocessing import OrdinalEncoder

# ordinal_columns = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4']
# oe = OrdinalEncoder()
# train_oe = oe.fit_transform(train[ordinal_columns])
# test_oe = oe.transform(test[ordinal_columns])

# ordinal_columns_oe = ['ord_0_oe_skl', 'ord_1_oe_skl', 'ord_2_oe_skl', 'ord_3_oe_skl', 'ord_4_oe_skl', 'ord_5_oe_skl']
# train = pd.concat([train, pd.DataFrame(train_oe, columns=ordinal_columns_oe)], axis=1)
# test = pd.concat([test, pd.DataFrame(test_oe, columns=ordinal_columns_oe)], axis=1)

# train.drop(ordinal_columns, axis=1, inplace=True)
# test.drop(ordinal_columns, axis=1, inplace=True)


# In[ ]:


# Check the data type of each feature.
train.dtypes


# In[ ]:


train.head()


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score as auc
from catboost import Pool, CatBoostClassifier
from category_encoders import TargetEncoder

# Specify the non-float features as categorical to the model.
categorical_features_indices = np.where(test.dtypes != np.float)[0]
print('Categorial Feature Indices: ', categorical_features_indices)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Model\ndef run_cv_model(categorical_indices, train, test, target, model_fn, params={}, eval_fn=None, label=\'model\', n_folds=5):\n    kf = KFold(n_splits=n_folds)\n    fold_splits = kf.split(train, target)\n    cv_scores = []\n    pred_full_test = 0\n    pred_train = np.zeros((train.shape[0]))\n    feature_importances = pd.DataFrame()\n    feature_importances[\'feature\'] = test.columns\n    i = 1\n    for dev_index, val_index in fold_splits:\n        print(\'-------------------------------------------\')\n        print(\'Started \' + label + \' fold \' + str(i) + f\'/{n_folds}\')\n        dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]\n        dev_y, val_y = target.iloc[dev_index], target.iloc[val_index]\n        params2 = params.copy()\n        pred_val_y, pred_test_y, fi = model_fn(categorical_indices, dev_X, dev_y, val_X, val_y, test, params2)\n        feature_importances[f\'fold_{i}\'] = fi\n        pred_full_test = pred_full_test + pred_test_y\n        pred_train[val_index] = pred_val_y\n        if eval_fn is not None:\n            cv_score = eval_fn(val_y, pred_val_y)\n            cv_scores.append(cv_score)\n            print(label + \' cv score {}: {}\'.format(i, cv_score), \'\\n\')\n        i += 1\n    print(\'{} cv scores : {}\'.format(label, cv_scores))\n    print(\'{} cv mean score : {}\'.format(label, np.mean(cv_scores)))\n    print(\'{} cv std score : {}\'.format(label, np.std(cv_scores)))\n    pred_full_test = pred_full_test / n_folds\n    results = {\'label\': label,\n              \'train\': pred_train, \'test\': pred_full_test,\n              \'cv\': cv_scores, \'fi\': feature_importances}\n    return results\n\n\ndef runCAT(categorical_indices, train_X, train_y, test_X, test_y, test_X2, params):\n    # Pool the data and specify the categorical feature indices\n    print(\'Pool Data\')\n    _train = Pool(train_X, label=train_y, cat_features = categorical_indices)\n    _valid = Pool(test_X, label=test_y, cat_features = categorical_indices)    \n    print(\'Train CAT\')\n    model = CatBoostClassifier(**params)\n    fit_model = model.fit(_train,\n                          eval_set=_valid,\n                          use_best_model=True,\n                          verbose=1000,\n                          plot=False)\n    feature_im = fit_model.feature_importances_\n    print(\'Predict 1/2\')\n    pred_test_y = fit_model.predict_proba(test_X)[:, 1]\n    print(\'Predict 2/2\')\n    pred_test_y2 = fit_model.predict_proba(test_X2)[:, 1]\n    return pred_test_y, pred_test_y2, feature_im\n\n\n# Use some baseline parameters\ncat_params = {\'loss_function\': \'CrossEntropy\', \n              \'eval_metric\': "AUC",\n              \'task_type\': "GPU",\n              \'learning_rate\': 0.01,\n              \'iterations\': 10000,\n              \'random_seed\': 42,\n              \'od_type\': "Iter",\n#               \'bagging_temperature\': 0.2,\n#               \'depth\': 10,\n              \'early_stopping_rounds\': 500,\n             }\n\nn_folds = 5\nresults = run_cv_model(categorical_features_indices, train, test, target, runCAT, cat_params, auc, \'cat\', n_folds=n_folds)')


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Make submission\nsubmission = pd.DataFrame({'id': test_id, 'target': results['test']})\nsubmission.to_csv('submission.csv', index=False)")


# In[ ]:


# Calculate the average feature importance for each feature
feature_importances = results['fi']
feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(n_folds)]].mean(axis=1)
feature_importances.to_csv('feature_importances.csv')
feature_importances.sort_values(by='average', ascending=False).head()


# In[ ]:


# Plot the feature importances with min/max/average using seaborn
feature_importances_flatten = pd.DataFrame()
for i in range(1, len(feature_importances.columns)-1):
    col = ['feature', feature_importances.columns.values[i]]
    feature_importances_flatten = pd.concat([feature_importances_flatten, feature_importances[col].rename(columns={f'fold_{i}': 'importance'})], axis=0)

plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importances_flatten.sort_values(by='importance', ascending=False), x='importance', y='feature')
plt.title('Feature Importances over {} folds'.format(n_folds))  
plt.savefig("feature_importances.png")

