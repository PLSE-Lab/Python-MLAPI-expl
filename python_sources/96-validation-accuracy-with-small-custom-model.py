#!/usr/bin/env python
# coding: utf-8

# As an aging but devout FRP fan, I just couldn't resist writing a quick and small model for this dataset. I previously optimized the model on my own machine using Hyperopt/Hyperas. 
# 
# Enjoy!

# In[ ]:


# import libraries
import os

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from keras import optimizers

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[ ]:


main_dir = os.listdir("../input")[0]
os.mkdir('./model_repo')


# In[ ]:


train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)


train_dir = os.path.join('../input', main_dir, 'dice' , 'train')
valid_dir = os.path.join('../input', main_dir, 'dice' , 'valid')
target_size = 150
batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(target_size, target_size),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(target_size, target_size),
    batch_size=batch_size,
    class_mode='categorical', 
    shuffle=False)


# In[ ]:



model = models.Sequential()
model.add(layers.Conv2D(16, 3, activation='relu', 
                        input_shape=(target_size, target_size, 3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(16, 5, activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(32, 5, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, 7, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(Dropout(0.6257491042113806))
model.add(layers.Dense(6, activation='softmax'))


# In[ ]:


model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
              optimizer=optimizers.RMSprop(3e-4))


# In[ ]:


callbacks_list = [
    callbacks.EarlyStopping(monitor='val_acc', 
                                 patience=7 
                                 ),
    callbacks.ModelCheckpoint(filepath='./model_repo/model.h5',
                                    monitor='val_loss',
                                    save_best_only=True
    ), 
    callbacks.ReduceLROnPlateau()
]


# In[ ]:


history = model.fit_generator(train_generator,
                             steps_per_epoch=int(train_generator.n // batch_size),
                             epochs=50,
                             verbose=1,
                             validation_data=validation_generator,
                             validation_steps=int(validation_generator.n // batch_size),
                             callbacks=callbacks_list
                             )


# In[ ]:



acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[ ]:


model = models.load_model('./model_repo/model.h5')


# In[ ]:


STEP_SIZE_EVAL = validation_generator.n//validation_generator.batch_size
validation_generator.reset()
preds = model.predict_generator(validation_generator, steps=STEP_SIZE_EVAL+1, verbose=1)


# In[ ]:


np.mean(np.argmax(preds, axis=1) == validation_generator.labels)


# In[ ]:


labels = (validation_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
labels = [i[1] for i in labels.items()]


# In[ ]:


confusion_matrix(validation_generator.labels, np.argmax(preds, axis=1))


# In[ ]:


print(classification_report(np.argmax(preds, axis=1), validation_generator.labels))


# In[ ]:


misclass_list = np.where(np.argmax(preds, axis=1) != validation_generator.labels)[0]

misclassed_files = np.array(validation_generator.filepaths)[misclass_list]


# In[ ]:


img = np.random.choice(misclassed_files)
image_data = plt.imread(img)
print(image_data.shape)
plt.imshow(image_data)


# Based on some limited sampling, most of the misclassified images seem like they are different sized images. Since the data is being resized to 150x150 there probably is a fair amount of distortion in the images. Getting it to a bigger size might solve the problem but then we might also need to enlarge the model which kinda defeats the purpose for this kernel.
# 
# Hope you enjoyed the kernel!
