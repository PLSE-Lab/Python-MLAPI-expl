#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
get_ipython().run_line_magic('matplotlib', 'inline')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


labeled_images = pd.read_csv('../input/digit-recognizer/train.csv')


# In[ ]:


labeled_images.info()


# In[ ]:


images = labeled_images.iloc[0:5000, 1:]
labels = labeled_images.iloc[0:5000, :1]
train_images, test_images, train_labels, test_labels = train_test_split(
images, labels, train_size=0.8, random_state=0) 


# In[ ]:


def show_images(i):
    img=train_images.iloc[i].values
    img=img.reshape((28,28))
    plt.imshow(img, cmap='gray')
    plt.title(train_labels.iloc[i,0])

show_images(0)


# In[ ]:


train_images.iloc[0].unique()


# In[ ]:


clf = svm.SVC(gamma='scale')
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images, test_labels)


# In[ ]:


imagest = labeled_images.iloc[5001:, 1:]
labelst = labeled_images.iloc[5001:, :1]


# In[ ]:


clf.score(imagest, labelst)


# In[ ]:


final_images = labeled_images.iloc[:, 1:]
final_labels = labeled_images.iloc[:, :1]
clff = svm.SVC(gamma='scale')
clff.fit(final_images, final_labels.values.ravel())


# In[ ]:


test = pd.read_csv('../input/digit-recognizer/test.csv')


# In[ ]:


predict = clff.predict(test)


# In[ ]:


df = pd.DataFrame(predict)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)

