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
import tensorflow as tf
from sklearn.model_selection import train_test_split
plt.style.use('fivethirtyeight')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df =  pd.read_csv('../input/Pokemon.csv')  #read the csv file and save it into a variable
df.head(n=10)      


# In[ ]:


#Here we going to do data preprocessing!

#We are going to use stats information, generation and legendary! so we exclude columns 0 through 3
X_train, X_test = train_test_split(df, 
                 test_size = 0.1)
input_shape = 9
print("Sets shape ", X_train.shape, X_test.shape)
print("Inputs ",input_shape)

logits = df['Type 1'].nunique()
print("Logits: ", logits)

y_train = X_train["Type 1"].values
y_test =  X_test["Type 1"].values

X_test_id =X_train.values[:, 0:2]
X_train_id = X_test.values[:, 0:2]

X_train = X_train.values[:, 4:]
X_test = X_test.values[:, 4:]

# super simple data pre process
scale = np.max(X_train)
X_train /= scale
X_test /= scale

#visualize scales
print("Max: {}".format(scale))

mapper_train, y_train_c = np.unique(y_train , return_inverse=True)
mapper_test, y_test_c = np.unique(y_test , return_inverse=True)

y_train_onehot = tf.keras.utils.to_categorical(y_train_c, num_classes=logits)
y_test_onehot = tf.keras.utils.to_categorical(y_test_c, num_classes=logits)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD


model = Sequential()
# Define DNN structure
model.add(Dense(32, input_dim=input_shape, activation='relu'))
model.add(Dense(units=logits, activation='softmax'))
     

model.compile(
    loss='categorical_crossentropy',
    optimizer='Adadelta',
    metrics=['accuracy']
    )
model.summary()


# In[ ]:




def gen_graph(history, title):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Accuracy ' + title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss ' + title)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# In[ ]:


history_rmsprop = model.fit(X_train, y_train_onehot,
          nb_epoch=500, validation_split = 0.1)



# In[ ]:


#plot the accuracy
gen_graph(history_rmsprop, 
              "ResNet50 RMSprop")


# In[ ]:


y_pred  = model.predict(X_test)

y_classes = y_pred.argmax(axis=-1)
preds_df = pd.DataFrame({'#': X_train_id[:,0], 'Name': X_train_id[:,1], "Type_predicted": mapper_train[y_classes[:]]})

preds_df.to_csv('pokemons_prediction.csv')
preds_df.head()

scores = model.evaluate(X_test, y_test_onehot, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

