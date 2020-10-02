#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing import image
from keras.applications import VGG16


# In[ ]:


base_dir = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/'
train_dir = os.path.join(base_dir, 'train/')
test_dir = os.path.join(base_dir, 'test/')


# In[ ]:


train_images = []
for i in os.listdir(train_dir):
    train_images.append(train_dir+i)
    
test_images = []
for i in os.listdir(test_dir):
    test_images.append(test_dir+i)


# In[ ]:


conv_m = VGG16(weights='imagenet',
               include_top=False,
               input_shape=(150,150,3))


# In[ ]:


conv_m.summary()


# In[ ]:


model = models.Sequential()
model.add(conv_m)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[ ]:


model.summary()


# In[ ]:


conv_m.trainable = False


# In[ ]:


model.compile(loss='binary_crossentropy',
             optimizer=optimizers.RMSprop(lr=2e-5),
             metrics=['acc'])


# In[ ]:


train = train_images[:2000]
train_numpy = []
for i in train:
    img = image.load_img(i, target_size=(150,150))
    img = image.img_to_array(img)
    img = img/255
    train_numpy.append(img)
    
train_numpy = np.array(train_numpy)


# In[ ]:


train_y = []
for i in train:
    if 'dog.' in i:
        train_y.append(1)
    else:
        train_y.append(0)


# In[ ]:


history = model.fit(train_numpy, train_y, batch_size=100, epochs=20, validation_split=0.25)


# In[ ]:


loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']


# In[ ]:


epochs = range(1, len(loss)+1)


# In[ ]:


plt.plot(epochs, loss, 'bo', label='Loss')
plt.plot(epochs, val_loss, 'b', label='Val_loss')
plt.title("Loss")
plt.legend()
plt.show()


# In[ ]:


plt.plot(epochs, acc, 'bo', label='Acc')
plt.plot(epochs, val_acc, 'b', label='Val_Acc')
plt.title("Acc")
plt.legend()
plt.show()


# In[ ]:


pred = model.predict(train_numpy[1:2])


# In[ ]:


if pred >= 0.5: 
    print('I am {:.2%} sure this is a Dog'.format(pred[0][0]))
else: 
    print('I am {:.2%} sure this is a Cat'.format((1-pred[0])[0]))


# In[ ]:


image.load_img(train[1])


# In[ ]:



test_numpy = []
for i in test_images:
    if '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test/test' in i :
        continue
    test_img = image.load_img(i, target_size=(150,150))
    test_img = image.img_to_array(test_img)
    test_img = test_img/255
    test_numpy.append(test_img)
    
test_numpy = np.array(test_numpy)


# In[ ]:


test_answer = model.predict(test_numpy)


# In[ ]:


test_id = []
for i in os.listdir(test_dir):
    if 'test' in i :
        continue    
    num = i.split('.')[0]
    test_id.append(num)


# In[ ]:


test_id_sub = pd.Series(test_id, name='id')


# In[ ]:


results = pd.Series(test_answer.reshape(12500,), name='label')


# In[ ]:


submission = pd.concat([test_id_sub, results], axis=1)


# In[ ]:


submission


# In[ ]:


submission.to_csv("Cats_and_Dogs_CNN_VGG.csv", index=False)

