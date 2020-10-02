#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/heart.csv')


# In[ ]:


df.head()


# Attribute Information:
# > 1. age
# > 2. sex
# > 3. chest pain type (4 values)
# > 4. resting blood pressure
# > 5. serum cholestoral in mg/dl
# > 6. fasting blood sugar > 120 mg/dl
# > 7. resting electrocardiographic results (values 0,1,2)
# > 8. maximum heart rate achieved
# > 9. exercise induced angina
# > 10. oldpeak = ST depression induced by exercise relative to rest
# > 11. the slope of the peak exercise ST segment
# > 12. number of major vessels (0-3) colored by flourosopy
# > 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
# 
# 

# In[ ]:


# stats
df.describe()


# In[ ]:


df.info()


# ## Observations:
# 1. 303 rows of data
# 2. population age: 29 -> 77 (avg 54)
# 3. Sex (=1) is 68.32 % of the population
# 4. Target (=1) is 54.46 % of the population
# 5. All features are encoded to class numbers
# 6. Most variables have a finite number of classes

# In[ ]:


plt.rcParams['figure.figsize'] = (20,8)
df.hist()


# In[ ]:


y = df['target']
df.drop('target', axis=1,inplace=True)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# In[ ]:


labelEncoder = LabelEncoder()
df['oldpeak'] = labelEncoder.fit_transform(y=df['oldpeak'])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=17)


# In[ ]:


tree = DecisionTreeClassifier(random_state=17, max_depth=3, min_samples_leaf=2)
tree.fit(X=X_train, y=y_train)


# In[ ]:


tree.score(X_test, y_test)


# In[ ]:


preds = tree.predict(X_test)
accuracy_score(y_true=y_test, y_pred=preds)


# Training a very basic DT classifier, we can see that it is possible to achieve a 75% accuracy. This is not a good performance, let's see if we can improve on this performance by using cross-validation.

# In[ ]:


export_graphviz(tree, 'tree1.dot', filled=True, feature_names=X_train.columns, rounded=True)
get_ipython().system("dot -Tpng 'tree1.dot' -o  'tree1.png'")


# In[ ]:


get_ipython().system('ls')


# <img src='tree1.png'>

# Let's use GridSearch and Stratified K-fold Cross-validation

# In[ ]:


from sklearn.model_selection import GridSearchCV, StratifiedKFold


# In[ ]:


# Let's vary hyperparameters from 2 - 10
best_parameters = {'max_depth': np.arange(2,11), 'min_samples_leaf': np.arange(2,11)}
decision_tree = DecisionTreeClassifier(criterion='entropy') # for information gain and entropy
model = GridSearchCV(estimator=decision_tree, param_grid=best_parameters, n_jobs=-1, verbose=1, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=17))
model.fit(X_train, y_train)
model.best_params_


# In[ ]:


model.best_score_


# In[ ]:


preds_2 = model.predict(X_test)


# In[ ]:


accuracy_score(y_test, preds_2)


# We can notice that we got a 8% improvement in the results when we use a grid search to find optimal hyperparameters and then use the derived tree to classify. So, let's visualize the tree again.

# In[ ]:


export_graphviz(model.best_estimator_, out_file='tree2.dot', filled=True, feature_names=X_train.columns, rounded=True)


# In[ ]:


get_ipython().system("dot -Tpng 'tree2.dot' -o 'tree2.png'")


# <img src='tree2.png'>

# Okay, so now, we have an accuracy of approximately 83%. Let's see if Logistic Regression can help.

# In[ ]:




