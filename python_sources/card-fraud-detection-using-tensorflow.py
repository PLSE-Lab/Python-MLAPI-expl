#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection using Tensorflow

# ## Import of Libaries and data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df =  pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')


# ## Quick EDA

# In[ ]:


df.describe().transpose()


# In[ ]:


df.info()


# In[ ]:


df['Class'].value_counts()


# In[ ]:


df.corr()['Class']


# In[ ]:


df= df.drop('Time', axis =1)


# ## Scaling the Data

# Due to imbalanced dataset we can reduce the split to train the model

# In[ ]:


fraud = df[df['Class'] == 1]
non_fraud = df[df['Class'] == 0]


# In[ ]:


non_fraud = non_fraud.sample(n = 1000)
non_fraud.shape


# In[ ]:


df = fraud.append(non_fraud)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# In[ ]:


X = df.drop('Class', axis = 1).values
y = df['Class'].values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[ ]:


scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# ## Model creatation

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


model = Sequential()
model.add(Dense(29,activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(15,activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(8,activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(units=1,activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy')


# In[ ]:


early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=5, patience=35)


# In[ ]:


model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          batch_size= 128,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )


# In[ ]:


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()


# ## Prediction and Evaluation

# In[ ]:


predictions = model.predict_classes(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))


# In[ ]:


print(confusion_matrix(y_test,predictions))

