#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import tensorflow.keras as keras
from keras import regularizers
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2
from keras.preprocessing.image import ImageDataGenerator

print(os.listdir("../input"))


# In[ ]:


df_train = pd.read_csv('../input/dog-breed-identification/labels.csv') 
df_test = pd.read_csv('../input/dog-breed-identification/sample_submission.csv') 


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


labels = df_train['breed']
one_hot = pd.get_dummies(labels, sparse = True)


# In[ ]:


one_hot_labels = np.asarray(one_hot)


# In[ ]:


im_resize = 64
#visualize a dogger
dogger1 = df_train['id'][0]
dogger2 = df_train['id'][1]
dogger3 = df_train['id'][2]
dogger4 = df_train['id'][3]
pupper1 = cv2.imread('../input/dog-breed-identification/train/{}.jpg'.format(dogger1))
pupper2 = cv2.imread('../input/dog-breed-identification/train/{}.jpg'.format(dogger2))
pupper3 = cv2.imread('../input/dog-breed-identification/train/{}.jpg'.format(dogger3), cv2.IMREAD_GRAYSCALE)
pupper4 = cv2.imread('../input/dog-breed-identification/train/{}.jpg'.format(dogger4), cv2.IMREAD_GRAYSCALE)
pupper4 = cv2.resize(pupper4, (im_resize, im_resize))
f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(pupper1)
axarr[0,1].imshow(pupper2)
axarr[1,0].imshow(pupper3,cmap="gray", vmin=0, vmax=255)
axarr[1,1].imshow(pupper4,cmap="gray", vmin=0, vmax=255)
plt.xticks([])
plt.yticks([])


# In[ ]:


im_size = pupper1.shape
print(im_size)
print(cv2.resize(pupper1, (im_resize, im_resize)).shape)


# In[ ]:


x_train = []
y_train = []
x_test = []


# In[ ]:


i = 0 
for f, breed in tqdm(df_train.values):
    img = cv2.imread('../input/dog-breed-identification/train/{}.jpg'.format(f))
    img_resized = cv2.resize(img, (im_resize, im_resize))
    x_train.append(img_resized)
    label = one_hot_labels[i]
    y_train.append(label)
    i += 1


# In[ ]:


del df_train


# In[ ]:


for f in tqdm(df_test['id'].values):
    img = cv2.imread('../input/dog-breed-identification/test/{}.jpg'.format(f))
    img_resized = cv2.resize(img, (im_resize, im_resize))
    x_test.append(img_resized)


# In[ ]:


num_class = 120 #static ftw


# In[ ]:


X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, shuffle=True,  test_size=0.1)


# In[ ]:


del x_train, y_train


# In[ ]:


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


# In[ ]:


datagen = ImageDataGenerator(width_shift_range=0.2,
                            height_shift_range=0.2,
                            zoom_range=0.2,
                            rotation_range=30,
                            vertical_flip=False,
                            horizontal_flip=True)


datagen.fit(X_train)


# In[ ]:



base_model = ResNet50(weights="../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",include_top=False, input_shape=(im_resize, im_resize, 3))

x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.25)(x)

logits = Dense(num_class, activation='softmax')(x)

model = Model(base_model.input, logits)

model.compile(optimizer='Adam',
          loss='categorical_crossentropy', 
           metrics=[categorical_crossentropy, categorical_accuracy])


# In[ ]:


def gen_graph(history, title):
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Accuracy ' + title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.plot(history.history['categorical_crossentropy'])
    plt.plot(history.history['val_categorical_crossentropy'])
    plt.title('Loss ' + title)
    plt.ylabel('MLogLoss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# In[ ]:


train_generator = datagen.flow(np.array(X_train), np.array(Y_train), 
                               batch_size=32) 


# In[ ]:


batch_size = 512

history_rmsprop = model.fit_generator(
    train_generator,
    epochs=30, steps_per_epoch=len(X_train) / batch_size,
    validation_data=(np.array(X_train), np.array(Y_train)), validation_steps=len(X_valid) / batch_size)


# In[ ]:


preds = model.predict(np.array(x_test), verbose=1)


# In[ ]:


#plot the accuracy
gen_graph(history_rmsprop, 
              "ResNet50 RMSprop")


# In[ ]:




sub = pd.DataFrame(preds)
col_names = one_hot.columns.values
sub.columns = col_names
sub.insert(0, 'id', df_test['id'])
sub.head(5)


# In[ ]:


sub.to_csv("output_rmsprop_aug.csv", index=False)

model.save('rmsprop_v2_augmentation.h5')

