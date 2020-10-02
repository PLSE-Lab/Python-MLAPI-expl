#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
import pandas
import random
import matplotlib.pyplot as plt
from keras import layers, models, optimizers
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Create Images Using CSV File and Extract Image Labels

# In[ ]:


os.chdir('../input/digit-recognizer')

def create_images(filename, mode='train'):
    dataset = pandas.read_csv(filename)
    images = []
    for i in range(0, dataset.shape[0]):
        if mode == 'train':
            pixels = dataset.iloc[i].tolist()[1:]
        else:
            pixels = dataset.iloc[i].tolist()
        img = np.zeros((28, 28))
        for j in range(0, len(pixels)):
            img[j // 28, j % 28] = pixels[j]
        images.append(img)
    if mode == 'train':
        return (np.asarray(dataset['label'].tolist()), np.asarray(images))
    else:
        return np.asarray(images)


# ### Loading the Training Dataset

# In[ ]:


train_labels, train_images = create_images('train.csv')


# In[ ]:


plt.figure(figsize=(20, 15))
for i in range(50):
    plt.subplot(5, 10, i + 1)
    plt.imshow(train_images[random.randrange(train_images.shape[0])])


# ### Reshape the Images for Keras

# In[ ]:


train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 1)


# ### Split Training and Test data

# In[ ]:


img_train, img_test, label_train, label_test = train_test_split(train_images, train_labels, test_size = 0.1)


# In[ ]:


train_images.shape[0] == img_train.shape[0] + img_test.shape[0]


# In[ ]:


train_labels.shape[0] == label_train.shape[0] + label_test.shape[0]


# ### CNN Model with Keras

# In[ ]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))  
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(layers.Dense(10, activation='softmax'))


# In[ ]:


model.summary()


# ### Data Augmentation and Compilation the Model

# In[ ]:


image_gen = ImageDataGenerator(rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               rescale=1./255,
                               shear_range=0.2,
                               zoom_range=0.2,
                               fill_mode='nearest')


# In[ ]:


train_datagen = image_gen.flow(img_train, label_train, batch_size=64)


# In[ ]:


model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])


# ### Fit the Model

# In[ ]:


history = model.fit_generator(train_datagen, steps_per_epoch=img_train.shape[0] // 64, 
                    epochs=20, verbose=1, callbacks=None, validation_data=(img_test, label_test))


# ### Accuracy and Loss Graph

# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(range(20), acc, 'r', label='Training accuracy')
plt.plot(range(20), val_acc, 'b', label='Validation accuracy')
plt.legend()
plt.show()


# In[ ]:


loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(range(20), loss, 'r', label='Training loss')
plt.plot(range(20), val_loss, 'b', label='Validation loss')
plt.legend()


# ### Predict

# In[ ]:


test_images = create_images('test.csv', 'test')


# In[ ]:


answer = model.predict(test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))


# In[ ]:


def classes(answer):
    numbers_classes = []
    for item in answer:
        numbers_classes.append(item.argmax())
    return numbers_classes


# In[ ]:


result = pandas.DataFrame({'ImageId': range(1, answer.shape[0] + 1), "Label": classes(answer)})


# In[ ]:


os.chdir('/kaggle/working')


# In[ ]:


result.to_csv('result.csv', index=False)


# In[ ]:


model.save('mnist.h5')

