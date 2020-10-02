#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/data.csv')
data.head()


# In[ ]:


data.describe()


# In[ ]:


data.isnull().sum()


# Apparently, there are 3 columns that should be dropped:
# 1. 'Unnamed: 32' -- Full of nulls
# 2. 'id' -- Sample ids which is irrelevant
# 3. 'diagnosis' -- Target column  needs to be binarized, stored in a variable and dropped as well.

# In[ ]:


data.dropna(axis=1, inplace=True)
data['diagnosis'] = data.diagnosis.apply(lambda x: 1 if x == 'M' else 0)
labels = data.diagnosis
data.drop(['id', 'diagnosis'], axis=1, inplace=True)


# In[ ]:


print('We have {} features in our data'.format(len(data.columns)))


# I will compare the 2 approaches: Gini & Entropy. I have provided some information about how they are calculated here: https://www.kaggle.com/ma7555/decision-trees-information-gain-from-scratch

# In[ ]:


# Splitting into train sets and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, 
                                                    train_size=0.8,
                                                    test_size=0.2)
# Classify

y_entropy = []
y_gini = []
for depth in range(len(data.columns)):
    classifier_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=depth+1)
    classifier_gini = DecisionTreeClassifier(criterion='gini', max_depth=depth+1)
    classifier_entropy.fit(x_train, y_train)
    classifier_gini.fit(x_train, y_train)
    y_entropy.append(classifier_entropy.score(x_test, y_test)*100)
    y_gini.append(classifier_gini.score(x_test, y_test)*100)
    
# print(classifier.tree_)


# In[ ]:


plt.figure(figsize=(12,4))
plt.plot(range(1, len(data.columns)+1), y_entropy)
plt.plot(range(1, len(data.columns)+1), y_gini)
plt.title('Gini Vs Entropy')
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.legend(['Entropy', 'Gini'])
plt.show()


# In[ ]:


best_accuracy = np.amax(y_entropy), np.amax(y_gini)
best_criterion = ['Entropy', 'Gini']

print('Best Criterion: {}, Accuracy {:.2f}% at depth = {}'.format(best_criterion[np.argmax(best_accuracy)], 
                                                                  np.amax(best_accuracy), 
                                                                  np.argmax(y_gini)+1 if np.amax(y_gini) > np.amax(y_entropy) else np.argmax(y_entropy)+1))


# A side note, please understand that depending on the random_state of selected test & train data, the output accuracy change!, sometimes gini is better, other times it is entropy.
