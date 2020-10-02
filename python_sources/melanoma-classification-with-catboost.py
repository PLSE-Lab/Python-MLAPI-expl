#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
test=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.columns


# In[ ]:


test.columns


# In[ ]:


train.head()


# In[ ]:


train = train.drop(['diagnosis','benign_malignant'], axis = 1)


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


car_feat = ['image_name', 'patient_id', 'sex', 'anatom_site_general_challenge']


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train['age_approx']=train['age_approx'].fillna((train['age_approx'].value_counts().index[0]))
train['sex']=train['sex'].fillna((train['sex'].value_counts().index[0]))


# In[ ]:


train.isnull().sum()


# In[ ]:


train['anatom_site_general_challenge']=train['anatom_site_general_challenge'].fillna((train['anatom_site_general_challenge'].value_counts().index[0]))
test['anatom_site_general_challenge']=test['anatom_site_general_challenge'].fillna((test['anatom_site_general_challenge'].value_counts().index[0]))


# In[ ]:


import seaborn as sns
sns.boxplot(x=train['age_approx'])


# In[ ]:


# replace outliar from age column
train['age_approx'] = train['age_approx'].replace(train['age_approx'].min(),train['age_approx'].median())


# In[ ]:


# seperate the features and target column
X = train.drop('target', axis=1)
y = train.target


# In[ ]:


# specify the categorical columns list
categorical_features_indices = np.where(X.dtypes != np.float)[0]


# In[ ]:


# split the training dataset into train and validation datasets
from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.85, random_state=42, stratify=y)

X_test = test


# In[ ]:


# import catboost library
from catboost import CatBoostClassifier, Pool, cv


# In[ ]:


model = CatBoostClassifier(
    eval_metric='AUC',
    random_seed=42,
    use_best_model=True,
    verbose=1  
)


# In[ ]:


model.fit(
    X_train, y_train,
    cat_features=categorical_features_indices,
    eval_set=(X_validation, y_validation),
#     logging_level='Verbose',  uncomment this for text output
    plot=False
);


# In[ ]:


# make prediction on validation dataset
predict = model.predict(X_validation)


# In[ ]:


# check AUC ROC score
from sklearn.metrics import roc_auc_score
score = roc_auc_score(y_validation, predict)
print('ROC AUC %.3f' % score)


# In[ ]:


# check the important features
train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
feature_importances = model.get_feature_importance(train_pool)
feature_names = X_train.columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print('{}: {}'.format(name, score))


# In[ ]:


predictions = model.predict_proba(X_test)[:,1]


# In[ ]:


sub = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
sub.head()


# In[ ]:


submission = pd.DataFrame({ 'image_name': test.image_name, 'target': predictions })
submission.to_csv('Submission_catboost.csv', index=False)


# In[ ]:




