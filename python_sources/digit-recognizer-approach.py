#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from sklearn.model_selection import train_test_split
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
train.head()


# In[ ]:


#convert first row into 1-D numpy array 
#matrix_nw = train.iloc[0].as_matrix()   //// as_matrix() can be removed in future so not using this
matrix_new = train.iloc[0].values
matrix_new = matrix_new[1:]
#matrix_new
#now reshape array in image size that is 28*28
matrix_new = matrix_new.reshape(28,28)
#visualize image
plt.imshow(matrix_new,cmap = 'gray')


# In[ ]:


# now split features and label for training purpose
y_train = train["label"]
x_train = train.drop(columns = {'label'})
training, validating,training_labels, validating_labels = train_test_split(x_train, y_train, train_size=0.8, random_state=0)


# In[ ]:


training.head()


# In[ ]:


training_labels.head()


# In[ ]:


import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[ ]:



classifiers = [
                
                SVC(),
               # LogisticRegression()
             ]


# In[ ]:


clf = SVC()
clf.fit(training, training_labels)


# In[ ]:


val_prdiction = clf.predict(validating)


# In[ ]:


acc = accuracy_score(validating_labels, val_prdiction)


# In[ ]:


acc


# In[ ]:


test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")


# In[ ]:


test_data


# In[ ]:


predicted = clf.predict(test_data)


# In[ ]:


type(predicted)


# In[ ]:


test_data["Label"] = predicted


# In[ ]:


x = [ i + 1 for i in range(len(test_data))]
test_data["ImageId"] = x


# In[ ]:


answer = test_data[["ImageId", "Label"]]
answer


# In[ ]:


answer.to_csv("result.csv",header=True,index=False)


# In[ ]:




