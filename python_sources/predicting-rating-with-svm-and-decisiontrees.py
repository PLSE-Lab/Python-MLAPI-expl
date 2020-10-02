#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Data input

# In[ ]:


df = pd.read_csv('/kaggle/input/goodreadsbooks/books.csv', error_bad_lines=False)
df.dropna(axis=0, how='any', inplace=True)
df.head()


# In[ ]:


print(df.shape)
df.describe()


# ## Data manipulation and visualization

# In[ ]:


# have to use ints for classification
df['average_rating_bin'] = pd.cut(df['average_rating'], 
                                  np.linspace(0, 5, 11), 
                                  labels=np.linspace(0, 9, 10)).fillna(0).astype(int)
df['# num_pages_bin'] = pd.cut(df['# num_pages'], 
                                  np.linspace(0, 7000, 29), 
                                  labels=np.linspace(0, 6750, 28)).fillna(0).astype(int)
df['average_rating_bin'].value_counts(), df['# num_pages_bin'].value_counts()


# In[ ]:


# average_ratingclustered between 3-5 stars, lower ratings will being noise.
# To make accurate predictions, limit data to ratings with 3-5 stars.

df = df[df['average_rating'] >= 3]

# have to use ints for classification
df['average_rating_bin'] = pd.cut(df['average_rating'], 
                                  np.linspace(3, 5, 11), 
                                  labels=np.linspace(0, 9, 10)).fillna(0).astype(int)


# In[ ]:


'''
average rating is as follows:
0 - [3.0, 3.2]
1 - [3.2, 3.4]
2 - [3.4, 3.6]
...
9 - [4.8, 5.0]
'''
sns.pairplot(df, hue='average_rating_bin')


# Almost all of the ratings are 3-5 stars. The majority are 3.6 - 4.4 stars.
# 
# Seems to be some relationship between number of pages and higher ratings. 
# 
# Ratings count and text reviews count seem to have a linear relationship. 
# 
# Nothing else seems to have a relationship.

# ## Analysis

# ### Lasso CV

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV

# ignore ID variables, titles and authors, and languages
X_cols = [x for x in df.columns if x not in ['bookID', 'title', 'authors', 
                                             'isbn', 'isbn13', 'language_code',
                                            'average_rating', 'average_rating_bin',
                                            '# num_pages_bin']]
print(X_cols)
df.dropna(subset=X_cols, axis=0, how='any', inplace=True)
X = df[X_cols]
y = df['average_rating_bin']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LassoCV().fit(X_train, y_train)

clf.alpha_, clf.coef_


# Only 3 variables so not much for LassoCV to exclude. Use all variables in final classification.

# ### Decision Tree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

params = {'max_depth': range(1, 5), 
          'max_features': range(1, 4), 
          'min_samples_leaf': np.arange(5, 26, 5)}

clf = GridSearchCV(DecisionTreeClassifier(), 
                   params,
                   iid=False,
                   n_jobs=1,
                   cv=5)
clf.fit(X_train, y_train)
tree_model = clf.best_estimator_

y_pred = tree_model.predict(X_test)

print(clf.best_score_, clf.best_params_, end='\n\n') # mean CV score
print(classification_report(y_test, y_pred)) # only for classification
print(confusion_matrix(y_test, y_pred)) # only for classification


# The model did not perform well with mean CV score of 30.20%.
# 
# Precision is the amount right vs the total amount, column-wise.
# 
# Recall is the amount right vs the total amount, row-wise.
# 
# F-score ranges b/w [0,1], with 1 being the best. Best value determines the strength of recall vs precision. When it equals 1, equal weights are given to precision and recall.
# 
# Bin [3.8, 4.0] had the best recall, and okay precision. It makes sense since it had the most observations as well. The data is so clustered in such a narrow range it makes any classification or prediction a challenge.

# ### SVM

# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

rbf_svc = SVC(kernel='rbf', decision_function_shape='ovo').fit(X_train, y_train)

y_pred = rbf_svc.predict(X_test)

print(classification_report(y_test, y_pred)) # only for classification
print(confusion_matrix(y_test, y_pred)) # only for classification
print(accuracy_score(y_test, y_pred)) # only for classification


# The results are even worse than the DecisionTreeClassifier with an accuracy score of only 29.03%.

# ## Conclusion

# The ratings are very clustered in a short range of 3.0 - 5.0 and even more so between 3.8 - 4.2. In addition, there doesn't appear to be a strong linear or non-linear relationship between ratings and any other variables. This makes prediction virtually impossible for the ratings variable as shown in the subpar performance of both the DecisionTreeClassifier and SVM.
