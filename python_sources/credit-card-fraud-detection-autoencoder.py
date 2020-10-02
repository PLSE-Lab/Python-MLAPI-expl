#!/usr/bin/env python
# coding: utf-8

# Link: https://www.kaggle.com/mlg-ulb/creditcardfraud

# In[ ]:


import tensorflow as tf
#tf.debugging.set_log_device_placement(True)
print(tf.__version__)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print('Found GPU at: {}'.format(device_name))

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense, Dropout


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
pd.options.display.float_format = "{:.4f}".format


# In[ ]:


print("shape:", df.shape)


# In[ ]:


print("columns: ", df.columns)


# In[ ]:


print("info: ", df.info())


# In[ ]:


print("describe: ", df.describe())


# In[ ]:


print("head: ")
print(df.head(10))


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[ ]:


X = df.iloc[:, 1:].values
X = sc.fit_transform(X)
print(X.shape)
print(X[0])


# In[ ]:


n_feature = X.shape[1]
print("n_feature: ", n_feature)


# In[ ]:





# In[ ]:


adam = tf.keras.optimizers.Adam(learning_rate=0.0001)

i = Input(shape=(n_feature,))
x = Dense(64, activation="relu")(i)
x = Dense(32, activation="relu")(x)
x = Dense(64, activation="relu")(x)
o = Dense(n_feature)(x)

model = Model(i,o)
model.compile(loss="mse", metrics=['accuracy'], optimizer=adam)
model.summary()


# In[ ]:


callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)
r = model.fit(X, X, epochs=500, batch_size=2064, verbose=1, validation_split=0.1, callbacks=[callback])


# In[ ]:


results = model.evaluate(X, X, batch_size=5, verbose=1)
print("Loss: %.2f" % results[0])
print("Acc: %.2f" % results[1])


# In[ ]:


print(r.history.keys())
plt.plot(r.history['loss'])
plt.plot(r.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()


# In[ ]:


plt.plot(r.history['accuracy'])
plt.plot(r.history['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'])
plt.show()


# In[ ]:


X_pred = model.predict(X)
mse = np.mean(np.power(X - X_pred, 2), axis=1)
mse


# In[ ]:


X_def = pd.DataFrame(X)


# In[ ]:


X_def['Loss_mae'] = mse
X_def.head(10)


# In[ ]:


#plt.figure()
fig, ax = plt.subplots()
import seaborn as sns
sns.set(color_codes=True)
g = sns.distplot(X_def['Loss_mae'],
             bins = 1, 
             kde= True,
            color = 'blue')
g.set(xlim=(0,0.1),ylim=(0,0.2))


# In[ ]:


X_def['Loss_mae'].describe()


# In[ ]:


mode_loss = X_def['Loss_mae'].mode()[0]
print("Mode Loss: ", mode_loss)
mean_loss = X_def['Loss_mae'].mean()
print("Mean Loss: ", mean_loss)
data_plt = g.get_lines()[0].get_data()
elbow = np.amax(data_plt[1])
t_loss_index = np.where(data_plt[1] == elbow)
t_loss = data_plt[0][t_loss_index][0]
print("Threshold Loss: ", t_loss)


# In[ ]:


X_def['Fraud'] = X_def['Loss_mae'] > t_loss
X_def['Fraud'] = X_def['Fraud'].apply(lambda x: 1 if x else 0)


# In[ ]:


X_def['old_class'] = df['Class']


# In[ ]:


X_def[X_def['Fraud'] != X_def['old_class']].head(50)


# In[ ]:


# predizioni errate classe esistente
X_def[X_def['old_class'] == 1]['Fraud'].value_counts()


# In[ ]:


pd.crosstab(X_def['Fraud'], X_def['old_class'])


# In[ ]:


cm = pd.crosstab(X_def['Fraud'], X_def['old_class'])
true_pos = np.sum(np.diag(cm))
false_pos = cm[0][1]
false_neg = cm[1][0]
#tot = np.sum(np.sum(cm, axis=0))
precision = true_pos / (true_pos + false_pos) * 100
recall = true_pos / (true_pos + false_neg) * 100
f1 = 2 * (precision * recall) / (precision + recall)
print("Precision: %.3f%%" % (precision))
print("Recall: %.3f%%" % (recall))
print("F1: %.3f%%" % (f1))

