#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.svm import SVC
from sklearn.decomposition import PCA


# In[ ]:


data_train = pd.read_csv('../input/train.csv')
x_test = pd.read_csv('../input/test.csv')


# In[ ]:


X_train = np.array(data_train[['pixel' + str(i) for i in range(0,784)]])
y_train = np.array(data_train['label'])
X_test = np.array(x_test[['pixel' + str(i) for i in range(0,784)]])


# In[ ]:


pca = PCA(n_components=30, whiten=True)


# In[ ]:


pca.fit(X_train)


# In[ ]:


x_train = pca.transform(X_train)
X_test = pca.transform(X_test)


# In[ ]:


clf = SVC()


# In[ ]:


clf.fit(x_train, y_train)


# In[ ]:


y = clf.predict(X_test)


# In[ ]:


submission = pd.DataFrame()
submission['ImageId'] = [i for i in range(1, len(x_test)+1)]
submission['Label'] = y


# In[ ]:


submission.to_csv('submission2.csv', index=False)

