#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv').astype('int16')
test_df = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv').astype('int16')
dig_mnist_df = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv').astype('int16')


# In[ ]:


X = train_df.drop(['label'],axis=1) / 255
y = train_df['label']

X_dig = dig_mnist_df.drop(['label'],axis=1) / 255
y_dig = dig_mnist_df['label']


# In[ ]:


## One-Hot Encode
y = keras.utils.to_categorical(y)
y_dig = keras.utils.to_categorical(y_dig)


# In[ ]:


model = MLPClassifier(verbose=1)
get_ipython().run_line_magic('time', 'model.fit(X,y)')


# In[ ]:


preds = model.predict(test_df.drop(['id'],axis=1))
preds = np.argmax(preds,axis=1,out=None)


# In[ ]:


submission = pd.DataFrame({
    'id' : test_df.id,
    'label' : preds
})

submission.to_csv('submission.csv',index=False)


# In[ ]:


submission


# In[ ]:




