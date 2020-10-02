#!/usr/bin/env python
# coding: utf-8

# 
# 

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
import matplotlib.pyplot as plt


# In[ ]:


img_rows, img_cols = 28, 28
num_classes = 10


# In[ ]:


def data_prep(raw):
    out_y = keras.utils.to_categorical(raw.label, num_classes)

    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y 


# In[ ]:


train_file = "../input/digit-recognizer/train.csv"
raw_data = pd.read_csv(train_file)
raw_data.sample(5)


# In[ ]:


pd.DataFrame(raw_data.values[:,150:190]).sample(5)


# In[ ]:


random_image_id = np.random.choice(raw_data.shape[0],1, replace=False)[0]
random_image_id
random_image_raw = raw_data.values[random_image_id,1:]
random_image_raw = random_image_raw.reshape(img_rows, img_cols)
plt.imshow(random_image_raw)
print(f'label={raw_data.values[random_image_id,0]}')


# In[ ]:


x, y = data_prep(raw_data)


# In[ ]:





# In[ ]:


raw = raw_data
out_y = keras.utils.to_categorical(raw.label, num_classes)
pd.DataFrame(out_y).head(10)

# raw.shape
# num_images = raw.shape[0]
# x_as_array = raw.values[:,1:]
# pd.DataFrame(x_as_array).sample(5)


# x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
# x_shaped_array
# out_x = x_shaped_array / 255
# out_x[40999][14][14][0]

# x = out_x
# y = out_y


# In[ ]:


model = Sequential()

# model.add(Dense(1, input_shape=(img_rows, img_cols, 1))) # first layer

model.add(Conv2D(20, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1))) # first layer

model.add(Conv2D(50, kernel_size=(4, 4), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))


model.add(Dense(num_classes, activation='softmax')) # last layer


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
model.summary()


# In[ ]:


model.fit(x, y,
          batch_size=128,
          epochs=3,
          validation_split = 0.2)


# 
# 
