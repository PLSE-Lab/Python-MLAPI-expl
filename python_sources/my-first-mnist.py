#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train=pd.read_csv("../input/mnist-in-csv/mnist_train.csv")
test=pd.read_csv("../input/mnist-in-csv/mnist_test.csv")


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


train.info()


# In[ ]:


test.head()


# In[ ]:


test.describe()


# In[ ]:


test.info()


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


print(train.columns)


# In[ ]:


print(np.sort(train['label'].unique()))


# In[ ]:


sns.countplot(train['label'])


# In[ ]:


sns.countplot(test['label'])


# In[ ]:


a=train.iloc[40,1:]             #Taking the 40th image in the training set
a.shape
dia=a.values.reshape(28,28)
dia.shape
plt.imshow(dia,cmap='gray')
plt.title('Digit')


# In[ ]:


# Separate the X and y variable
X=train.drop(columns='label',axis=1)
print(X.shape)

y=train['label']
print(y.shape)

X_test=test.drop(columns='label',axis=1)
print(X_test.shape)

y_test=test['label']
print(y_test.shape)


# In[ ]:


#NORMALISATION

X=X/255.0
y=y/255.0


# In[ ]:


from sklearn.preprocessing import scale
X=scale(X)
X_test=scale(X_test)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, train_size = 0.2 ,random_state = 10)



from sklearn.preprocessing import LabelEncoder
lab_enc =LabelEncoder()
y_train = lab_enc.fit_transform(y_train)


# In[ ]:


from sklearn.svm import SVC
model=SVC(kernel='rbf')
model.fit(X_train,y_train)


# In[ ]:


y_predict=model.predict(X_test)
print(y_predict)


# In[ ]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix

accuracy=metrics.mean_squared_error(y_test,y_predict)
print(accuracy)


# In[ ]:


fig=X_train[650,:]
fig=fig.reshape(28,28)
plt.imshow(fig,cmap='gray')
print(y_predict[650])


# In[ ]:





# In[ ]:




