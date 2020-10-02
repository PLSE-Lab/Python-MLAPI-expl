#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import norm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()


# In[ ]:


train.set_index('id', drop=True, inplace=True)
target = train.target
train.drop('target', axis=1, inplace=True)
test.set_index('id', drop=True, inplace=True)


# In[ ]:


print(train.dtypes)


# Almost all columns are float64, let's see if any is not actually float.

# In[ ]:


for column in train.columns:
    if train[column].dtype != np.float64:
        print(column, train[column].dtype)


# In[ ]:


train['wheezy-copper-turtle-magic'].describe()


# This is a column, which as many has figured out is probably not relevant to the data, but rathen an index.
# If has values from 0 to 511, only integers. One idea is to create 512 distinct classifier, and depending on wheezy-copper-turtle-magic value, you would use the classifier trained for it. If this provided better accuracy scores than ignoring this value all together and using 1 classifier only then our initial idea could possibly be correct.
# 
# I will plot all other float columns on a kdeplot just to see how the data looks like and if there is any significant other columns that we should consider.

# In[ ]:


float_features = list(train.columns)
float_features.remove('wheezy-copper-turtle-magic')

plt.figure(figsize=(10,4))
for feature in float_features:
    g = sns.kdeplot(train[feature], shade=True)
    g.legend().set_visible(False)
plt.show()


# All the features, more or less, appear to have similar distribution. Gaussian or not?
# We can try to plot a perfect Gaussian having the same mean and std to decide.

# In[ ]:


mu = np.mean(train[float_features[0]])
sigma = np.std(train[float_features[0]])

x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
plt.plot(x, norm.pdf(x, mu, sigma))
sns.kdeplot(train[float_features[0]])
plt.show()


# So no, they are not Gaussian indeed. They are much taller and thinner than Gaussian distribution having same mean and standard deviation. 
# Does this really matter for our model? That depends on which classifier/kernel you decided to choose. Some kernels/classifiers are designed to work best with Gaussian data, others don't really care (decsion trees as an example)

# In[ ]:


# Feature selection -- To be implemented later on
features = train.columns

# Splitting into train sets and test sets
x_train, x_test, y_train, y_test = train_test_split(train[features], target, 
                                                    train_size=0.8, test_size=0.2,
                                                    random_state=555)


# # The Naive Model:
# I will just drop the turtle magic column, create a single classifier, and see the results. I will approach it using SVM and Decsion Trees for now. 
# Given the number of rows, it would be impractical to use the RBF kernel on the whole data as training time will be very large.
# We can however explore the possibility of using the Nystroem transformer or poly kernel directly if we see any hope.
# I would just use the first 2.5K rows to see how things are going.

# In[ ]:


classifier = SVC(kernel = 'rbf')
classifier.fit(x_train[:2500].drop('wheezy-copper-turtle-magic', axis=1), y_train[:2500])
print(classifier.score(x_test.drop('wheezy-copper-turtle-magic', axis=1), y_test))


# In[ ]:


classifier = DecisionTreeClassifier(criterion='entropy', random_state=555)
classifier.fit(x_train[:2500].drop('wheezy-copper-turtle-magic', axis=1), y_train[:2500])
print(classifier.score(x_test.drop('wheezy-copper-turtle-magic', axis=1), y_test))


# It is obvious that things are pretty much going nowhere using this approach... The data could indeed be meaningless if combined together and turtle-magic is the secret column in this competition. So, Let's try and train distinct 512 classifier!

# # A Corrected Approach:

# In[ ]:


train.groupby('wheezy-copper-turtle-magic').size()


# In[ ]:


len(train)/512


# So we have 512 subsets of around 512 rows per subset.

# In[ ]:


# Classify using turtle-magic as index for datasets without any parameter tuning...
score_trees = []
score_svm = []

for i in range(512):
    sub_train = x_train[x_train['wheezy-copper-turtle-magic'] == i]
    sub_target_train = target.loc[sub_train.index]
    sub_test = x_test[x_test['wheezy-copper-turtle-magic'] == i]
    sub_target_test = target.loc[sub_test.index]
    
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=555)
    classifier.fit(sub_train, sub_target_train)
    score_trees.append(classifier.score(sub_test, sub_target_test))
    
    classifier = SVC(kernel = 'rbf')
    classifier.fit(sub_train, sub_target_train)
    score_svm.append(classifier.score(sub_test, sub_target_test))
    if i % 40 == 0: print('Completed: {:.1f}%'.format(i*100/512))
print('Trees:', np.mean(score_trees))
print('SVM:', np.mean(score_svm))


# The SVM score looks promising! We still did not try to tune the gamma or C parameters or use any feature selection.
# 

# In[ ]:


# Feature Selection, Grid Search with cross-validation for every turtle-magic
# This will take a  long time.. 
score_svm = []
best_c = []
best_gamma = []
best_features_mask = []

for i in range(512):
    sub_train = train[train['wheezy-copper-turtle-magic'] == i]
    sub_target_train = target.loc[sub_train.index]
    
    sel = VarianceThreshold(threshold=3) # returns around 40 features +- ..
    
    sub_train = sel.fit_transform(sub_train)
    best_features_mask.append(sel.get_support())

    C_range = [0.75, 1.5, 3]
    gamma_range = [0.0012, 0.0016, 0.002, 0.0024, 0.0028, 0.0032, 0.0036]
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.075)
    grid = GridSearchCV(SVC(kernel = 'rbf'), param_grid=param_grid, cv=cv)
    grid.fit(sub_train, sub_target_train)

    best_c.append(grid.best_params_['C'])
    best_gamma.append(grid.best_params_['gamma'])

    if i % 15 == 0: print("Best parameters for turtle-magic {}: (C: {} & gamma: {}) - Best score of {:.3f} - Picked n_features = {}".format(i, grid.best_params_['C'], grid.best_params_['gamma'], grid.best_score_, sub_train.shape[1]))
    if i % 15 == 0: print('Completed: {:.1f}%'.format(i*100/512))


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')
submission.set_index('id', drop=False, inplace=True)
submission.head()


# In[ ]:


for i in range(512):

    sub_train = train[train['wheezy-copper-turtle-magic'] == i]
    sub_target_train = target.loc[sub_train.index]
    
    sub_test = test[test['wheezy-copper-turtle-magic'] == i]
    
    sub_features = sub_train.columns[best_features_mask[i]]
    
    classifier = SVC(kernel = 'rbf', C=best_c[i], gamma=best_gamma[i])
    classifier.fit(sub_train[sub_features], sub_target_train)
    
    prediction = classifier.predict(sub_test[sub_features])
    df = pd.DataFrame({'id': sub_test.index, 'target': prediction})
    df.set_index('id', inplace=True)
    submission.update(df)
    if i % 40 == 0: print('Completed: {:.1f}%'.format(i*100/512))
submission.head(20)


# In[ ]:


submission.to_csv('submission.csv', index=False)

