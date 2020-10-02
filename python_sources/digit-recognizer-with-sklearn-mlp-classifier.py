#!/usr/bin/env python
# coding: utf-8

# # **Learning sklearn MLP Classifier**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# # **Taking Input of dataset**

# In[ ]:


train_data=pd.read_csv("../input/train.csv")


# # **Preview of our dataset**

# In[ ]:


train_data.head()


# # **Converting data to matrix**

# In[ ]:


train_data=train_data.as_matrix()


# # **Preparing data by seperating labels and features**

# In[ ]:


train_label=train_data[:,0]


# In[ ]:


train_label


# In[ ]:


train_features=train_data[:,1:]


# In[ ]:


print(train_features)
np.shape(train_features)


# In[ ]:


from sklearn.neural_network import MLPClassifier


# # **Taking part of the dataset to make Test set**

# In[ ]:


test_label=train_label[30001:]


# In[ ]:


test_features=train_features[30001:,:]


# In[ ]:


train_label=train_label[:30000]


# In[ ]:


train_features=train_features[:30000,:]
## decreasing the values to get them between 0 and 1
train_features=train_features/255


# # **Initializing our MLP Classifier**

# In[ ]:


clf=MLPClassifier(solver='adam',hidden_layer_sizes=350,alpha=1e-04)


# In[ ]:


clf.fit(train_features,train_label)


# # **Getting Score on our Test set**

# In[ ]:


clf.score(test_features,test_label)


# In[ ]:




