#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
import tensorflow as tf
import os

plt.style.use('ggplot')


# In[2]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/sample_submission.csv')


# In[3]:


train['has_cactus'] = train['has_cactus'].astype('str')
train['has_cactus'].value_counts()


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


train_df, valid_df = train_test_split(train, test_size=0.1, stratify=train['has_cactus'], random_state=42)


# In[6]:


rows, cols = (2, 5)

fig, ax = plt.subplots(rows,cols,figsize=(20,5))

for j in range(rows):
    for i, sample in enumerate(train_df[j * cols:rows * cols - (cols * (rows - (j + 1)))].values):
        path = os.path.join('../input/train/train', sample[0])
        ax[j][i].imshow(img.imread(path))
        ax[j][i].set_title('Label: ' + str(sample[1]))
        ax[j][i].grid(False)
        ax[j][i].set_xticklabels([])
        ax[j][i].set_yticklabels([])


# In[7]:


datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255., 
                                                                vertical_flip=True, horizontal_flip=True,)
datagen_valid = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


# In[8]:


img_size = 224


# In[9]:


train_data = datagen_train.flow_from_dataframe(dataframe=train_df, directory='../input/train/train',
                                               x_col='id', y_col='has_cactus', batch_size=64,
                                               class_mode='binary', target_size=(img_size, img_size))


validation_data = datagen_valid.flow_from_dataframe(dataframe=valid_df,directory='../input/train/train',
                                                    x_col='id', y_col='has_cactus', batch_size=64,
                                                    class_mode='binary', target_size=(img_size, img_size))


# In[10]:


model_vgg16 = tf.keras.applications.VGG16(include_top=False, input_shape=(img_size, img_size, 3))


# In[11]:


for layer in model_vgg16.layers:
    layer.trainable = False


# In[12]:


leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.3)
leaky_relu.__name__ = 'leaky_relu'
flat1 = tf.keras.layers.Flatten()(model_vgg16.layers[-1].output)
class1 = tf.keras.layers.Dense(256, activation=leaky_relu)(flat1)
drop1 = tf.keras.layers.Dropout(0.5)(class1)
class2 = tf.keras.layers.Dense(256, activation=leaky_relu)(drop1)
drop2 = tf.keras.layers.Dropout(0.5)(class2)
output = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)

model_vgg16 = tf.keras.models.Model(inputs=model_vgg16.inputs, outputs=output)


# In[13]:


model_vgg16.summary()


# In[14]:


adadelta = tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
model_vgg16.compile(optimizer=adadelta, loss='binary_crossentropy', metrics=['accuracy'])


# In[15]:


checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("my_best_model.h5",
                                                   save_best_only=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                                     restore_best_weights=True)


# In[16]:


history = model_vgg16.fit(train_data, epochs=100,
                          validation_data=validation_data, 
                          callbacks=[checkpoint_cb, early_stopping_cb])


# In[17]:


history_df = pd.DataFrame(history.history)
history_df.plot(figsize=(13, 10))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


# In[18]:


test_data = datagen_valid.flow_from_dataframe(dataframe=test, directory="../input/test/test",
                                              x_col="id", y_col=None, shuffle=False, 
                                              class_mode=None, target_size=(img_size, img_size))


# In[19]:


answer = pd.DataFrame({'id': test['id']})


# In[20]:


answer['has_cactus'] = model_vgg16.predict(test_data, verbose=True)


# In[21]:


answer.head()


# In[22]:


answer.to_csv('submission.csv',  sep=',' , line_terminator='\n', index=False)


# In[ ]:




