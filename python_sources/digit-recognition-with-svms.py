#!/usr/bin/env python
# coding: utf-8

# ## Data Loading

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Data Exploration
# 
# As this is the first notebook of the series, let's explore the data structures given.

# In[ ]:


_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
_train.head(3)


# In[ ]:


print(f"Unique digit classes for labelling: {np.sort(pd.unique(_train['label']))}")
print("{} {} {} {} {} {} {} {} {} {}".format(*list(range(0,10))))


# In[ ]:


print("Digit distribution over the train set")
_train["label"].value_counts()


# In[ ]:


import seaborn as sns

sns.countplot(_train["label"])
plt.show()


# In[ ]:


sample_digit = np.array(_train.iloc[1, 1:])  # we've to drop the first value, which is the label
print("Original shape:", sample_digit.shape)

sample_digit = sample_digit.reshape((28,28))

plt.imshow(sample_digit, interpolation='nearest', cmap="gray")
plt.show()


# In[ ]:


_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
_test.head(3)


# In[ ]:


_output = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
print(f"Number of labels: {pd.unique(_output['Label'])}")
_output.head(3)


# As we can see, data is provided as one-hot encoded. As for submission, we've to pair each Image ID with a label or class.
# 
# <hr/>
# 
# Let's start with our Data Exploration and implementation of Support Vector Machines.
# 
# ## Modelling with Support Vector Machines
# 
# 

# In[ ]:


# https://scikit-learn.org/stable/modules/svm.html
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

_train_X = _train.iloc[:, 1:]
_train_Y = _train.iloc[:, 0]

X_train, X_test, Y_train, Y_test = train_test_split(
    _train_X,
    _train_Y,
    test_size=.2,
    random_state=30
)


# In[ ]:


default_pipe = [("scaler", StandardScaler())]
poly_pipe = default_pipe + [("SVM", svm.SVC(kernel="poly"))]

pipe = Pipeline(poly_pipe)
pipe


# In[ ]:


search_parameters = dict(SVM_C=[.001, .01, 100, 10e5], SVM_gamma=[10, 1, .1, .01])
grid = GridSearchCV(pipe, param_grid=search_parameters, cv=5)
grid


# In[ ]:


grid.fit(X_train, Y_train)

print("Score =", grid.score(X_test, Y_test))
print("Best parameters from train data:", grid.best_params_)

Y_pred = grid.predict(X_test)


# If we already know the best parameters, we can just run the SVM kernel

# In[ ]:


poly_pipe = default_pipe + [("SVM", svm.SVC(kernel="poly", C=0.001, gamma=10.0))]

final_pipe = Pipeline(poly_pipe)


# In[ ]:


# final_pipe.fit(X_train, Y_train)

print("Score =", final_pipe.score(X_test, Y_test))

# Y_pred = final_pipe.predict(X_test)

hits = np.sum(Y_pred == Y_test)

print(f"Correct label predictions: {hits} out of {len(Y_test)} ({hits / len(Y_test)})")


# In[ ]:


# using subplots
# https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib

# fig, subarr = plt.subplots(2,3)
# sub_map = {0: (0,0), 1: (0,1), 2:(0,2), 3:(1,0), 4: (1,1), 5: (1,2)}

for i, pos in enumerate(np.random.randint(0, Y_pred.shape[0], 6)):
    digit = np.reshape(X_test.values[pos], (28,28)) * 255
    digit = digit.astype(np.uint8)
    
    plt.imshow(digit, interpolation="nearest", cmap="gray")
    plt.title(f"Predicted label: {Y_pred[pos]}")
    plt.show()

