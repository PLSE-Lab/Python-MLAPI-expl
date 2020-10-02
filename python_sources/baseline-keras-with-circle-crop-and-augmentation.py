#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tqdm import tqdm

from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')


# In[ ]:


IMG_SIZE = 224


# In[ ]:


def load_img(path):
    image = cv2.imread(path)
    return image


# Let us take a look at some of the images.

# In[ ]:


fig, axes = plt.subplots(3,3,figsize=(10,10))
selection = np.random.choice(train_df.index, size=9, replace=False)
images = '../input/aptos2019-blindness-detection/train_images/'+train_df.loc[selection]['id_code']+'.png'
for image, axis in zip(images, axes.ravel()):
    img = load_img(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axis.imshow(img)


# As we can see, there are extra black areas surrounding the eyes which can be removed. The lighting conditions also differ significantly from image to image.

# Let's crop images so that the extra spaces surrounding eyes are removed.

# In[ ]:


def remove_unwanted_space(image, threshold=7):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray_image > threshold
    return image[np.ix_(mask.any(1), mask.any(0))]


# Ben Graham's preprocessing method (last competition) is used for fixing lighting conditions. Please refer to this kernel for more: https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping

# In[ ]:


def preprocess_img(path):
    image = load_img(path)
    image = remove_unwanted_space(image, 5)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image,4, cv2.GaussianBlur(image, (0,0), 30), -4, 128)
    return image


# In[ ]:


fig, axes = plt.subplots(3,3,figsize=(10,10))
selection = np.random.choice(train_df.index, size=9, replace=False)
images = '../input/aptos2019-blindness-detection/train_images/'+train_df.loc[selection]['id_code']+'.png'
for image, axis in zip(images, axes.ravel()):
    img = preprocess_img(image)
    axis.imshow(img)


# There seems a correlation between the way images are cropped in the training data and the target variables. In order to avoid the model predicting targets based on the way images are cropped, we can circle crop the eyes ourself in preprocessing. Refer to this kernel for more info: https://www.kaggle.com/taindow/be-careful-what-you-train-on 

# In[ ]:


def circle_crop(img):
    circle_img = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
    cv2.circle(circle_img, ((int)(IMG_SIZE/2),(int)(IMG_SIZE/2)), int(IMG_SIZE/2), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    return img


# In[ ]:


fig, axes = plt.subplots(3,3,figsize=(10,10))
selection = np.random.choice(train_df.index, size=9, replace=False)
images = '../input/aptos2019-blindness-detection/train_images/'+train_df.loc[selection]['id_code']+'.png'
for image, axis in zip(images, axes.ravel()):
    img = circle_crop(preprocess_img(image))
    axis.imshow(img)


# In[ ]:


N = train_df.shape[0]
train = np.empty((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
for i, image_id in enumerate(tqdm(train_df['id_code'])):
    train[i,:,:,:] = circle_crop(preprocess_img('../input/aptos2019-blindness-detection/train_images/'+image_id+'.png'))


# In[ ]:


N = test_df.shape[0]
test = np.empty((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
for i, image_id in enumerate(tqdm(test_df['id_code'])):
    test[i,:,:,:] = circle_crop(preprocess_img('../input/aptos2019-blindness-detection/test_images/'+image_id+'.png'))


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(train, train_df['diagnosis'], test_size=0.15, random_state=42)


# In[ ]:


BATCH_SIZE = 32


# In[ ]:


train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        zoom_range=[0.9, 1.0],
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
        rotation_range=120
    )


# In[ ]:


val_data_generator = tf.keras.preprocessing.image.ImageDataGenerator()


# In[ ]:


train_gen = train_data_generator.flow(X_train, y_train, batch_size=BATCH_SIZE)


# In[ ]:


val_gen = val_data_generator.flow(X_val, y_val, batch_size=BATCH_SIZE)


# We can consider this problem as a regression problem instead of classification. Here targets 0,1,2,3,4 are different stages of diabetic retinopathy and are not just independent classes. Their magnitude is representative of the severity of the disease. So this is a regression problem.

# In[ ]:


resnet = tf.keras.applications.ResNet50(include_top=False, weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape=(IMG_SIZE, IMG_SIZE, 3))


# In[ ]:


model = tf.keras.Sequential([
    resnet,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='relu')
])


# In[ ]:


checkpoint = tf.keras.callbacks.ModelCheckpoint('model_weights.hdf5', monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min')


# In[ ]:


optimizer = tf.keras.optimizers.Adam(lr=0.00005)


# In[ ]:


model.compile(optimizer=optimizer, loss='mse')


# In[ ]:


steps_per_epoch = int(np.ceil(X_train.shape[0]/BATCH_SIZE))
val_steps_per_epoch = int(np.ceil(X_val.shape[0]/BATCH_SIZE))


# In[ ]:


history = model.fit(train_gen, validation_data=val_gen, steps_per_epoch=steps_per_epoch, 
                              validation_steps=val_steps_per_epoch, callbacks=[checkpoint], epochs=25)


# In[ ]:


model.load_weights('model_weights.hdf5')


# In[ ]:


prediction = model.predict(test)


# In[ ]:


for i, pred in enumerate(prediction):
    if pred < 0.5:
        prediction[i] = 0
    elif pred < 1.5:
        prediction[i] = 1
    elif pred < 2.5:
        prediction[i] = 2
    elif pred < 3.5:
        prediction[i] = 3
    else:
        prediction[i] = 4


# In[ ]:


prediction = np.squeeze(prediction.astype(np.int8))


# In[ ]:


sample = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
sample.diagnosis = prediction
sample.to_csv("submission.csv", index=False)

