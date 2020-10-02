#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from sklearn.datasets import load_iris
iris = load_iris()


# In[ ]:


X = iris.data
y = iris.target


# In[ ]:


from keras.utils import to_categorical
y = to_categorical(y)
y.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler_object = MinMaxScaler()
scaler_object.fit(X_train)


# In[ ]:


scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


model.fit(scaled_X_train, y_train, epochs=150, verbose=2)


# In[ ]:


y_pred = model.predict_classes(scaled_X_test)


# In[ ]:


y_test.argmax(axis=1)


# In[ ]:


from sklearn import metrics

cm = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Confusion Matrix - '+str(metrics.accuracy_score(y_test.argmax(axis=1),y_pred))
plt.title(all_sample_title, size = 15);
plt.show()
print(metrics.classification_report(y_test.argmax(axis=1),y_pred))
print('------------------------------')


# In[ ]:





# In[ ]:




