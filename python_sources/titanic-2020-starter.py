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


# # DataFrame

# In[ ]:


train_df = pd.read_csv('/kaggle/input/1056lab-titanic-survivors-prediction-2020/train.csv', index_col=0)
test_df = pd.read_csv('/kaggle/input/1056lab-titanic-survivors-prediction-2020/test.csv', index_col=0)


# In[ ]:


train_df


# In[ ]:


test_df


# # Visualizing

# ## Scatterplot matrix

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
sns.pairplot(train_df)
plt.show()


# ## Correlation heatmap

# In[ ]:


corr = train_df.corr()

plt.figure()
sns.heatmap(corr, square=True, annot=True)
plt.show()


# # Preprocessing

# ## Feature selection

# In[ ]:


train_df = train_df.drop('Name', axis=1)
test_df = test_df.drop('Name', axis=1)


# ## Replacing string to numeric

# In[ ]:


train_df['Sex'] = train_df['Sex'].map({'male':0, 'female':1})
test_df['Sex'] = test_df['Sex'].map({'male':0, 'female':1})


# In[ ]:


train_df


# In[ ]:


test_df


# # NumPy data

# In[ ]:


X_train = train_df.drop('Survived', axis=1).to_numpy()
y_train = train_df['Survived'].to_numpy()
X_test = test_df.to_numpy()


# In[ ]:


X_train


# In[ ]:


y_train


# In[ ]:


X_test


# # Learning a decision tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)


# # Visualizing the decision tree

# In[ ]:


get_ipython().system(' pip3 install dtreeviz')


# In[ ]:


from dtreeviz.trees import dtreeviz

viz = dtreeviz(model, X_train, y_train, target_name='Survived',
               feature_names=test_df.columns, class_names={0:'no', 1:'yes'})
viz


# # Predicting

# ## Training data

# In[ ]:


p_train = model.predict_proba(X_train)
p_train


# ## Test data

# In[ ]:


p_test = model.predict_proba(X_test)
p_test


# # Making a submission file

# In[ ]:


submit_df = pd.read_csv('/kaggle/input/1056lab-titanic-survivors-prediction-2020/sampleSubmission.csv', index_col=0)
submit_df['Survived'] = p_test[:,1]
submit_df


# In[ ]:


submit_df.to_csv('submission.csv')

