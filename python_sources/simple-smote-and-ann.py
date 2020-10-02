#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
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


df = pd.read_csv('../input/creditcardfraud/creditcard.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


df.columns


# Normalized

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


df['Amount(Normalized)'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))


# In[ ]:


df.loc[:,['Amount', 'Amount(Normalized)']].head()


# In[ ]:


df = df.drop(columns = ['Amount', 'Time'], axis=1)


# In[ ]:


X = df.drop('Class', axis=1)


# In[ ]:


y = df['Class']


# SMOTE Sampling to deal with imbalanced data

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:


X_smote, y_smote = SMOTE().fit_sample(X, y)


# In[ ]:


X_smote = pd.DataFrame(X_smote)
y_smote = pd.DataFrame(y_smote)


# In[ ]:


y_smote.iloc[:,0].value_counts()


# Artificial Neural Networks

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=0)


# In[ ]:


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout


# In[ ]:


model = Sequential([
    Dense(units=20, input_dim = X_train.shape[1], activation='relu'),
    Dense(units=24, activation='relu'),
    Dropout(0.5),
    Dense(units=20, activation='relu'),
    Dense(units=24, activation='relu'),
    Dense(1, activation='sigmoid')
])


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=30, epochs=5)


# In[ ]:


score = model.evaluate(X_test, y_test)
print('Test Accuracy: {:.2f}%\nTest Loss: {}'.format(score[1]*100,score[0]))


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


y_pred = model.predict(X)
y_test = pd.DataFrame(y)
cm2 = confusion_matrix(y_test, y_pred.round())
sns.heatmap(cm2, annot=True, fmt='.0f')
plt.show()


# In[ ]:


score_new = model.evaluate(X, y)
print('Test Accuracy: {:.2f}%\nTest Loss: {}'.format(score_new[1]*100, score_new[0]))


# In[ ]:


print(classification_report(y_test, y_pred.round()))


# In[ ]:


Thank you!

