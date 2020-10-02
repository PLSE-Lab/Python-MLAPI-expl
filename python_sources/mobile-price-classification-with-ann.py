#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('../input/mobile-price-classification/train.csv')


# In[ ]:


df.head(10)


# In[ ]:


df.info()


# In[ ]:


print(np.unique(df['price_range']))


# In[ ]:


df.describe()


# In[ ]:


# Lets Divide the data to input and target
X = df.iloc[:,0:20]
y = df.iloc[:,-1]


# In[ ]:


# Lets do some Feature Selection for better results and accuracy
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest


# In[ ]:


kbest = SelectKBest(chi2,k=10)


# In[ ]:


best_feaures = kbest.fit(X,y)


# In[ ]:


best_feaures.scores_


# In[ ]:


df_features = pd.DataFrame(best_feaures.scores_)
df_columns = pd.DataFrame(X.columns)


# In[ ]:


featureScores = pd.concat([df_columns,df_features],axis=1)


# In[ ]:


featureScores.columns = ['Features','Score']


# In[ ]:


featureScores.sort_values(by='Score',ascending=False)


# In[ ]:


X = df[['ram','px_height','battery_power','px_width','mobile_wt','int_memory','sc_w','talk_time','fc','sc_h']]


# In[ ]:


X


# In[ ]:


X = X.values
y = y.values


# In[ ]:


print(X.shape,y.shape)


# In[ ]:


# lets do some normalisation and scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# In[ ]:


y = y.reshape(-1,1)


# In[ ]:


# now convert y labels to one hot encoder
from sklearn.preprocessing import OneHotEncoder
ohot = OneHotEncoder()
y = ohot.fit_transform(y)


# In[ ]:


y = y.toarray()


# In[ ]:


# Train Test Split
from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)


# In[ ]:


# Neural network Model
import keras
from keras.models import Sequential
from keras.layers import *
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


model = Sequential()
model.add(Dense(8,activation='relu',input_dim = 10))
model.add(Dense(6,activation='relu'))
model.add(Dense(4,activation='softmax'))
model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


# In[ ]:


history = model.fit(X_train,y_train,epochs=105,validation_data=(X_test,y_test),batch_size=64)


# In[ ]:


y_pred = model.predict(X_test)

#lets do the inverse one hot encoding
pred = []
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
    
# also inverse encoding for y_test labels

test = []
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))


# In[ ]:


# accuracy of the model
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred,test)
print("Accuracy of Your Model is = " + str(acc*100))


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Features','Price_Weight'],loc='upper left')
plt.show()


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Features','Price_Weight'],loc='upper left')
plt.show()


# ### Please Upvote if you like the Notebook!
# ### ThankYou!
