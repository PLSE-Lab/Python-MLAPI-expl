#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


ds = pd.read_csv('/kaggle/input/forbes-celebrity-100-since-2005/forbes_celebrity_100.csv')
ds


# In[ ]:


ds.info()


# top 20 celebrities

# In[ ]:


ds = ds.set_index('Name')
ds_top_20_celeb = ds['Pay (USD millions)'].nlargest(20)
ds_top_20_celeb


# In[ ]:


# mean salary of top 20 celebrities
ds_top_20_celeb.mean()


# In[ ]:


ds_top_20_celeb.plot(kind = 'bar')


# In[ ]:


df_grup_by_Category = ds.groupby(["Category","Name"])["Pay (USD millions)"].mean()
df_grup_by_Category


# In[ ]:


ds['Category'].unique()

creating dummies of category
# In[ ]:


ds_with_dummies = pd.get_dummies(ds.Category)
ds_with_dummies.head()


# concating the dummies with actual dataset and dropping unnecessary columns

# In[ ]:


ds = pd.concat([ds, ds_with_dummies], axis=1)
ds


# In[ ]:


ds.drop('Category', axis=1, inplace=True)
ds.head()


# In[ ]:


ds.info()


# In[ ]:


x = ds.drop(ds_with_dummies.columns, axis=1)
y = ds['Authors']
x.head()


# In[ ]:


y.value_counts()


# splitting dataset into train and test 

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


# In[ ]:


dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt.score(x_test,y_test)


# In[ ]:


xg = XGBClassifier()
xg.fit(x_train,y_train)
xg.score(x_test,y_test)


# In[ ]:


rf = RandomForestClassifier(n_estimators=25)
rf.fit(x_train,y_train)
rf.score(x_test,y_test)


# In[ ]:


kn = KNeighborsClassifier()
kn.fit(x_train,y_train)
kn.score(x_test,y_test)


# In[ ]:


svm = SVC()
svm.fit(x_train,y_train)
svm.score(x_test,y_test)


# # Confusion Matrix

# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix

y_pred1 = svm.predict(x_test)
cm = confusion_matrix(y_test,y_pred1)
print('Confusion matrix\n',cm)


# In[ ]:


plt.figure(figsize=(7,5))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('truth')


# # **Standardizing data for CNN**

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[ ]:


y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# In[ ]:


import keras
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

monitor = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, restore_best_weights=True)


# In[ ]:


x_train_cnn = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test_cnn = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

x_train_cnn.shape


# # **CNN**

# In[ ]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(8, kernel_size=2, activation='relu', input_shape = (2,1)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train_cnn,y_train, epochs=10, validation_split=0.2, callbacks=[monitor])


# # Learning curves

# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuacy')
plt.legend(['Acc','Val'], loc = 'upper left')


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['loss','Val'], loc = 'upper left')


# In[ ]:


y_pred = model.predict_classes(x_test_cnn)
acc = accuracy_score(y_test,y_pred)
print('Accuracy : ',acc)


# In[ ]:


cm = confusion_matrix(y_test,y_pred)
print('Confusion matrix\n',cm)


# In[ ]:


plt.figure(figsize=(7,5))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('truth')

