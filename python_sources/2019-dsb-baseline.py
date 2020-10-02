#!/usr/bin/env python
# coding: utf-8

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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_context('talk')


# In[ ]:


def read_data():
    print(f'Read data')
    train_df = pd.read_csv('../input/data-science-bowl-2019/train.csv')
    test_df = pd.read_csv('../input/data-science-bowl-2019/test.csv')
    train_labels_df = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
    specs_df = pd.read_csv('../input/data-science-bowl-2019/specs.csv')
    sample_submission_df = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
    
    return train_df, test_df, train_labels_df, specs_df, sample_submission_df


# In[ ]:


train_df, test_df, train_labels_df, specs_df, sample_submission_df = read_data()


# In[ ]:


train_df.head()


# In[ ]:


train_labels_summary = (
    train_labels_df
    .groupby(['accuracy_group'])
    .agg('count')
    .rename(columns={'title': 'count'})
    .sort_values(by='count', ascending=False)
)

(
    train_labels_summary
    .plot
    .bar(y='count')
)


# In[ ]:


total_count = train_labels_df.shape[0]
train_labels_summary['prob'] = train_labels_summary['count'] / total_count
(
    train_labels_summary
    .plot
    .bar(y='prob')
)


# In[ ]:


sample_submission_df.head()


# In[ ]:


from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate, train_test_split


# In[ ]:


train_data = pd.merge(
    train_df, 
    train_labels_df, 
    how='right',
    on=[
        'game_session',
        'installation_id'
    ]
)


# In[ ]:


train_data.head()


# In[ ]:


np.random.seed(42)
train_data_grouped = train_data.groupby([
    'installation_id',
    'accuracy_group'
]).agg('count').reset_index()
X_train, X_test, y_train, y_test = train_test_split(
    train_data_grouped[['installation_id']],
    train_data_grouped['accuracy_group']
)


# In[ ]:


np.random.seed(42)
cls = DummyClassifier()
cls.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(
    y_train,
    cls.predict(X_train)
))


# In[ ]:


print(classification_report(
    y_test,
    cls.predict(X_test)
))


# In[ ]:


from sklearn.metrics import cohen_kappa_score


# In[ ]:



print('Cohen kappa score for train set: {:.5f}'.format(cohen_kappa_score(
    y_train,
    cls.predict(X_train)
)))


# In[ ]:



print('Cohen kappa score for validation set: {:.5f}'.format(cohen_kappa_score(
    y_test,
    cls.predict(X_test)
)))


# In[ ]:


test_df_grouped = test_df.groupby([
    'installation_id'
]).agg('count').reset_index()
submission_df = pd.DataFrame(dict(
    installation_id=test_df_grouped['installation_id'],
    accuracy_group=cls.predict(test_df_grouped[['installation_id']])
))


# In[ ]:


submission_df.to_csv('submission.csv', index=False)

