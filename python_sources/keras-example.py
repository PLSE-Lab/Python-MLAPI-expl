#!/usr/bin/env python
# coding: utf-8

# # Esempio normalizzazione DataSet

# Verifichiamo l'efficacia della normalizzazione dei Dati in questo esempio tratto da Kaggle [Kaggle competition](https://www.kaggle.com/c/forest-cover-type-prediction).
# 
# Forest Cover Type Prediction<br>
# Use cartographic variables to classify forest categories
# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/3936/logos/front_page.png)

# In[ ]:


#Import pandas, tensorflow e keras
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.python.data import Dataset
import keras
from keras import regularizers
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import backend as K

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#   for filename in filenames:
#       print(os.path.join(dirname, filename))
#Lettura dati
df = pd.read_csv("/kaggle/input/forest-cover-type-prediction/train.csv")
dfT = pd.read_csv("/kaggle/input/forest-cover-type-prediction/test.csv")


# In[ ]:


#Selezioniamo le caratteristiche
x = df[df.columns[1:55]]
xT = dfT[dfT.columns[1:55]]
#Selezioniamo le etichette (8) 
y = df.Cover_Type
#Split data into train and test 
x_train, x_test, y_train, y_test = train_test_split(x, y , train_size = 0.7, random_state =  90)


# ## Normalizziamo la rete come da teoria
# \begin{equation}
#   x^{(i)} = x^{(i)}-\frac{1}{m}\sum_{i=1}^{m} x^{(i)} \\
#   x^{(i)} = \frac{x^{(i)}}{\frac{1}{m}\sum_{i=1}^{m} {x^{(i)}}^2}
# \end{equation}

# In[ ]:


# Normalize Training Data 
scaler = preprocessing.StandardScaler()
scaler.fit(x_train.values[:,0:10])
x_train_norm = scaler.transform(x_train.values[:,0:10])
x_test_norm = scaler.transform(x_test.values[:,0:10])
x_sub = scaler.transform(xT.values[:,0:10])
x_train_norm=numpy.concatenate((x_train_norm,x_train.values[:,10:]),axis=1)
x_test_norm=numpy.concatenate((x_test_norm,x_test.values[:,10:]),axis=1)
x_sub=numpy.concatenate((x_sub,xT.values[:,10:]),axis=1)


# ### Creiamo una nuova funzione di regolarizzazione da testare ;-) 

# In[ ]:


def l0_reg(weight_matrix):
    temp = K.abs(weight_matrix)>0.005
    if_true = tf.reduce_sum(tf.cast(temp, tf.float32))
    return if_true


# > ## Creiamo la rete Neurale con Keras con Doppio regolarizzatore

# Creiamo una rete a 5 Livelli in cui abbiamo definito l'architettura, il tipo di inizializzazione dei parametri.

# In[ ]:


modelF = models.Sequential()
modelF.add(layers.Dense(32,name="Layer_1",activation='relu',input_dim=54,kernel_initializer='he_normal',kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.08)))
modelF.add(layers.BatchNormalization())
modelF.add(layers.Dense(16,name="Layer_2",activation='relu'))
modelF.add(layers.Dense(64,name="Layer_22",activation='relu'))
modelF.add(layers.BatchNormalization())
modelF.add(layers.Dense(64,name="Layer_23",activation='relu'))
modelF.add(layers.BatchNormalization())
modelF.add(layers.Dense(16,name="Layer_4",activation='relu'))
modelF.add(layers.Dense(8,name="Layer_5",activation='softmax'))
modelF.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
modelF.summary()


# In[ ]:


Net4 = modelF.fit(
 x_train_norm, y_train,
 epochs= 400, batch_size = 256,
 validation_data = (x_test_norm, y_test))


# Grafichiamo delle curve per la valutazione

# In[ ]:


_, train_acc = modelF.evaluate(x_train_norm, y_train, verbose=0)
_, test_acc = modelF.evaluate(x_test_norm, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss during training
plt.rcParams['figure.figsize'] = (12.0, 9.0)
plt.subplot(211)
plt.title('Loss')
plt.plot(Net4.history['loss'], label='train')
plt.plot(Net4.history['val_loss'], label='test')
plt.legend()
# plot accuracy during training
plt.subplot(212)
plt.title('Accuracy')
plt.plot(Net4.history['acc'], label='train')
plt.plot(Net4.history['val_acc'], label='test')
plt.legend()
plt.show()


# In[ ]:


test_predictions=modelF.predict_classes(x_sub, batch_size=256, verbose=0)


# In[ ]:


solutions = pd.DataFrame({'Id':dfT.Id, 'Cover_Type':test_predictions})
solutions.to_csv('submission.csv',index=False)


# In[ ]:




