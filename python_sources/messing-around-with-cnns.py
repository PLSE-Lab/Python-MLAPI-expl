#!/usr/bin/env python
# coding: utf-8

# In[76]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder
import os
print(os.listdir("../input"))


# In[77]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[78]:


train_pixels = train_df.iloc[:, 1:].values / 255.0
train_labels = train_df.iloc[:, 0].values
test_pixels = test_df.values / 255.0


# In[79]:


print(train_pixels.shape)
print(train_labels.shape)


# In[80]:


train_pixels = train_pixels.reshape([-1, 28, 28])
validation_size = int(len(train_pixels) * 0.1)
validation_pixels = train_pixels[-validation_size:]
train_pixels = train_pixels[:-validation_size]
test_pixels = test_pixels.reshape([-1, 28, 28])
encoder = OneHotEncoder(10)
train_labels = train_labels.reshape([-1,1])
labels = encoder.fit_transform(train_labels).toarray()
validation_labels = labels[-validation_size:]
labels = labels[:-validation_size]
print(train_pixels.shape)
print(validation_pixels.shape)
print(test_pixels.shape)
print(labels.shape)
print(validation_labels.shape)


# In[81]:


import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[82]:


plt.figure()
plt.imshow(train_pixels[69], cmap='gray')
plt.figure()
plt.imshow(train_pixels[420], cmap='gray')
plt.figure()
plt.imshow(train_pixels[666], cmap='gray')


# In[83]:


from keras.layers import Input, Dropout, Conv2D, Flatten, Dense, MaxPooling2D, Reshape, BatchNormalization
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator


# In[84]:


input_tensor = Input((28,28, 1))
net = Conv2D(16, (3,3), padding='same', activation='relu')(input_tensor)
net = BatchNormalization()(net)
net = Conv2D(16, (3,3), padding='same', activation='relu')(net)
net = MaxPooling2D()(net)
net = BatchNormalization()(net)
net = Conv2D(32, (3,3), padding='same', activation='relu')(net)
net = BatchNormalization()(net)
net = Conv2D(32, (3,3), padding='same', activation='relu')(net)
net = MaxPooling2D()(net)
net = BatchNormalization()(net)
net = Flatten()(net)
net = Dense(512, activation='relu')(net)
net = BatchNormalization()(net)
# net = Dense(2048, activation="relu")(net)
# net = BatchNormalization()(net)
output_logits = Dense(10, activation='softmax')(net)
model = Model(inputs=input_tensor, outputs=output_logits)


# In[85]:


model.summary()


# In[86]:


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5)


# In[87]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[88]:


train_pixels = train_pixels.reshape([-1,28,28,1])
validation_pixels = validation_pixels.reshape([-1,28,28,1])
test_pixels = test_pixels.reshape([-1, 28,28,1])
generator = ImageDataGenerator(rotation_range=10,shear_range=10,zoom_range=0.1, height_shift_range=0.2, width_shift_range=0.2)
generator.fit(train_pixels)
print(train_labels.shape)
print(labels.shape)


# In[89]:


# model.fit(train_pixels, labels, batch_size=128,epochs=12,validation_split=0.02, callbacks=[reduce_lr])
model.fit_generator(generator.flow(train_pixels, labels, batch_size=128),steps_per_epoch=len(train_pixels) / 128 + 29, epochs=100, validation_data=(validation_pixels, validation_labels), callbacks=[reduce_lr])


# In[ ]:


test_pixels = test_pixels.reshape([-1, 28, 28, 1])
with open('submission.csv', "w") as file:
    file.write("ImageId,Label\n")
    for idx in range(len(test_pixels)):
        if (idx + 1) % 100 == 0:
            print ("{}/{}".format(idx + 1, len(test_pixels)))
        result = model.predict(np.array([test_pixels[idx]]))[0]
        file.write("{},{}\n".format(idx+1, np.argmax(result)))


# In[ ]:




