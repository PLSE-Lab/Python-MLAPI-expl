#!/usr/bin/env python
# coding: utf-8

# Trying the model explained here
# 
# https://github.com/mk-gurucharan/Face-Mask-Detection/blob/master/FaceMask-Detection.ipynb

# ## DL imports 

# In[ ]:


import os
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
import numpy as np


# In[ ]:


#from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


# In[ ]:


# image_size = 150 
# train_data_generator = ImageDataGenerator(
#                                     rotation_range=40,
#                                    width_shift_range=0.2,
#                                    height_shift_range=0.2,
#                                    shear_range=0.2,
#                                    zoom_range=0.2,
#                                    horizontal_flip=True)


# train_generator = train_data_generator.flow_from_directory(
#         '../input/prajna-bhandary-face-mask-detection-dataset/data/train/',
#         target_size=(image_size, image_size),
#         batch_size= 10)

# validation_data_generator = ImageDataGenerator()
# validation_generator = validation_data_generator.flow_from_directory(
#         '../input/prajna-bhandary-face-mask-detection-dataset/data/test/',
#         target_size=(image_size, image_size))


# from tensorflow.python.keras.applications.resnet50 import preprocess_input
# 
# I have seen in the DL from scratch section that you can supply 
# 
# ImageDataGenerator(preprocessing_function=preprocess_input)

# In[ ]:


from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.applications.resnet import preprocess_input


# In[ ]:


def img_path(path):
    for dirname, _, filenames in os.walk(path):
        l = []
        y = []
        for filename in filenames:
            l.append(os.path.join(dirname, filename))
            y.append(os.path.join(dirname, filename).split("/")[-2])
        return (l, y)

image_size = 150
def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(output)


# In[ ]:


l_img_path, out_y = img_path("../input/prajna-bhandary-face-mask-detection-dataset/data/train/with_mask/")
x, y = img_path("../input/prajna-bhandary-face-mask-detection-dataset/data/train/without_mask/")
l_img_path = l_img_path + x 
out_y = out_y + y 
x, y = img_path("../input/prajna-bhandary-face-mask-detection-dataset/data/test/with_mask/")
l_img_path = l_img_path + x 
out_y = out_y + y 
x, y = img_path("../input/prajna-bhandary-face-mask-detection-dataset/data/test/without_mask/")
l_img_path = l_img_path + x 
out_y = out_y + y 
#print(l_img_path)
#print(out_y)
out_x = read_and_prep_images(l_img_path)

for i in range(len(out_y)):
    if out_y[i] == "with_mask":
        out_y[i] = 1
    else:
        out_y[i] = 0
from tensorflow import keras
# print(out_y)
out_y = keras.utils.to_categorical(out_y, 2)


# In[ ]:


print(out_y)


# In[ ]:


model = Sequential([
    Conv2D(30, (3,3), activation='relu', input_shape=(150, 150, 3)),
    Conv2D(30, (3,3), activation='relu'),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss= keras.losses.categorical_crossentropy,  metrics=['acc'])


# In[ ]:


from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')


# In[ ]:


# history = model.fit_generator(train_generator,
#                               epochs=3,
#                               validation_data=validation_generator,
#                               callbacks=[checkpoint])
hist = model.fit(out_x, out_y,
          batch_size=10,
          epochs=2)


# In[ ]:


# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from PIL import Image
# test_img_path = '../input/prajna-bhandary-face-mask-detection-dataset/data/train/without_mask/10.jpg'
# image = Image.open(test_img_path)
# new_image = image.resize((150, 150))
# plt.imshow(new_image)
# mat = np.array(new_image)
# result=model.predict(mat)


# In[ ]:


img_paths = ['../input/prajna-bhandary-face-mask-detection-dataset/data/train/with_mask/1-with-mask.jpg', '../input/prajna-bhandary-face-mask-detection-dataset/data/train/without_mask/10.jpg', '../input/prajna-bhandary-face-mask-detection-dataset/data/train/without_mask/108.jpg'] 
test_img = read_and_prep_images(img_paths)
preds = model.predict(test_img)
print(preds)


# with mask means 0 1 
# 
# wihout mask means 1 0 

# In[ ]:


I have not added even the validation set i will add after 

