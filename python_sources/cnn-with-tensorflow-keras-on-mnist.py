#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Nanangk/CNN_with_TensorFlow-Keras/blob/master/CNN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# #Convolutional Neural Network With TenserFlow + Keras

# Import Library yang dibutuhkan

# In[ ]:


import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import TensorBoard


# In[ ]:


#Download datanya
(xtrain, ytrain),(xtest, ytest) = mnist.load_data()


# Mari kita cek Jumlah data dan sizenya

# In[ ]:


xtrain.shape


# In[ ]:


xtest.shape


# Terdapat 60.000 Gambar data training dan 1000 gambar data testing dengan size 28 x 28 piksel

# In[ ]:


#coba cek gambarnya
plt.imshow(xtest[3000])


# In[ ]:


ytest[3000]


# In[ ]:


#cek ytrain masih berupa numerical
ytrain[3000]


# In[ ]:


xtrain = xtrain.reshape(60000, 28,28,1)
xtest = xtest.reshape(10000,28,28,1)


# In[ ]:


#konversi ytrain dan ytest dari numerical ke ctegorical
ytrain = keras.utils.to_categorical(ytrain, 10)
ytest = keras.utils.to_categorical(ytest, 10)


# Buat Model CNN nya

# In[ ]:





# In[ ]:


model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), activation='linear', padding='same', input_shape=(28,28,1)))
model.add(Conv2D(64, (3,3), activation='linear', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='linear'))
model.add(Dropout(0.25))
model.add(Dense(10, activation=Activation(tf.nn.softmax)))


# Liat Bentuk Modelnya

# In[ ]:


model.summary()


# Compile Modelnya

# In[ ]:


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])


# In[ ]:


#Tensorboard utk melihat grafik proses training model nanti
callbacks = TensorBoard(log_dir='./Graph')


# Waktunya Mentraining

# In[ ]:


model.fit(xtrain, ytrain, 
          batch_size=128, 
          epochs=30, 
          verbose=1,
          validation_data=(xtest, ytest),
          callbacks=[callbacks])


# Coba Predict

# In[ ]:


datatest = xtest[650]
xd = model.predict(datatest.reshape(1,28,28,1))
xd=xd[0]
xs=np.arange(0,10,1)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(20,7))
ax.bar(xs,xd)
plt.show()

a_max = xd.max()
mylist = xd
i=0
for i in range(len(mylist)):
  if mylist[i] == a_max:
    print(str(a_max * 100)+ '% menunjukkan ini angka :' +str(i))
  i+=1
   


# In[ ]:


#mau nyari posisi angka di data test
def cari(x):
  i=0
  for i in range(len(ytest)):
    if x == np.argmax(ytest[i], axis=-1) :
      print("ada di index "+str(i))
      break  

cari(8)


# In[ ]:




