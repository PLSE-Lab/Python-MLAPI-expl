#!/usr/bin/env python
# coding: utf-8

# **This NotebooK is A startter code for Cat boost which uses is Encoding power and Hold out Validation Strategy. Upvote more to come **

# In[ ]:





# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd #for data importing and manupulation
import numpy as np  #for data manupulation and cleaning
import seaborn as sns # data visualization
import matplotlib.pyplot as plt # data visualization
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')
test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')
sub = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')


# In[ ]:


from catboost import CatBoostClassifier


# In[ ]:





# In[ ]:


#Imputing missing values for both train and test
train.fillna(0, inplace=True)
test.fillna(0,inplace=True)


# In[ ]:


sub.head()


# In[ ]:


test_id = test['id']


# In[ ]:


train = train.drop('id', axis=1)
test = test.drop('id', axis=1)


# In[ ]:





# In[ ]:


X = train.drop('target', axis=1)
y = train.target


# In[ ]:


len(test)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


# Split the data into 30% validation and 70% training
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=101)


# In[ ]:


categorical_features_indices = np.where(X.dtypes != np.float)[0]


# In[ ]:


model=CatBoostClassifier(iterations=1000, #leaf_estimation_iterations=10,#800
                              learning_rate=0.1,
                              depth=5,#loss_function='MultiClass',#,scale_pos_weight=200,
                             #l2_leaf_reg=10,
                             bootstrap_type='Bernoulli',
                              subsample=0.9,
                              eval_metric='AUC',
                              metric_period=20,
                                   #class_weight s=0,
                              #od_type='Iter',
                              #od_wait=45,
                              #random_seed=10,
                              allow_writing_files=False)


# In[ ]:


model.fit(X_train, y_train, cat_features=categorical_features_indices, eval_set=(X_val, y_val))


# In[ ]:


y_pred=model.predict_proba(test)[:, 1]


# In[ ]:


from sklearn.metrics import roc_auc_score


# In[ ]:


print(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))


# In[ ]:


fea_imp = pd.DataFrame({'imp':model.feature_importances_, 'col': X.columns})
fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
_ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))
plt.savefig('catboost_feature_importance.png')    


# In[ ]:


d = {"id": test_id, 'target': y_pred.round(3)}
test_predictions = pd.DataFrame(data=d)
test_predictions = test_predictions[["id", 'target']]


# In[ ]:


test_predictions.head()


# In[ ]:


test_predictions.to_csv('kagle_cat2.csv', index=False)


# In[ ]:




