#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install scikit-plot')


# In[ ]:



from glob import glob
import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils import np_utils
import scikitplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dropout, Dense, Flatten,GlobalMaxPool2D,GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from keras.layers import Input
from keras.utils import plot_model
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau


# In[ ]:


os.listdir('../input/ck48-5-emotions/CK+48')


# In[ ]:


INPUT_PATH = "../input/ck48-5-emotions/CK+48/"
total_images = 0
for dir in os.listdir(INPUT_PATH):
    count = len(os.listdir(INPUT_PATH + dir))
    total_images += count
    print(f"{dir} has {count} images")

print(f"total_images : {total_images}")


# In[ ]:


TOP_EMOTIONS = ['anger', 'surprise', 'happy']
# total_images
# img_arr = np.empty(shape=(total_images, 48, 48, 1))
# img_label = np.empty(shape=(total_images))
# label_to_text = {}

# idx = 0
# label = 0
# for dir_ in os.listdir(INPUT_PATH):
#     if dir_ in  TOP_EMOTIONS:
#         for f in os.listdir(INPUT_PATH + dir_ + "/"):
#             img_arr[idx] = np.expand_dims(cv2.imread(INPUT_PATH + dir_ + "/" + f, 0), axis=2)
#             img_label[idx] = label
#             idx += 1
#         label_to_text[label] = dir_
#         label += 1

# img_label = np_utils.to_categorical(img_label)


e=0
i=0
img_label=[]
img_arr=[]
label_to_text = {}

for dir_ in os.listdir(INPUT_PATH):
    if dir_ in TOP_EMOTIONS:
        label_to_text[e] = dir_
        for f in os.listdir(INPUT_PATH + dir_ + "/"):
            img_arr.append( np.array(cv2.imread(INPUT_PATH + dir_ + "/" + f)))
            img_label.append(e)
            i+=1
        print(f"loaded all {dir_} images to numpy arrays")  
        e+=1
        
img_arr,img_label = np.array(img_arr),np.array(img_label)
print(img_arr.shape,img_label.shape)

img_label = np_utils.to_categorical(img_label)
img_label.shape


# In[ ]:


img_arr = img_arr / 255.

X_train, X_test, y_train, y_test = train_test_split(img_arr, img_label,
                                                    shuffle=True, stratify=img_label,
                                                    train_size=0.7, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


img_width = X_train.shape[1]
img_height = X_train.shape[2]
img_depth = X_train.shape[3]
num_classes = y_train.shape[1]


# In[ ]:


model= ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(img_width, img_height, img_depth)))
# model.summary()


# In[ ]:


plot_model(model, show_shapes=True, show_layer_names=True, expand_nested=True, dpi=50)


# In[ ]:


# x = model.layers[-5].output
# global_pool = GlobalMaxPool2D(name="global_pool")(x)
# predictions = Dense(num_classes, activation="softmax")(global_pool)

# model = Model(inputs=model.input, outputs=predictions)


x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x) 
predictions = Dense(3, activation='softmax')(x)
model = Model(model.input, predictions)


# In[ ]:


plot_model(model, show_shapes=True, show_layer_names=True, expand_nested=True, dpi=50, )


# In[ ]:


model.layers


# In[ ]:


for layer in model.layers[:38]:
    layer.trainable = False


# In[ ]:


trainAug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

valAug = ImageDataGenerator()


# In[ ]:


trainAug.fit(X_train)


# In[ ]:


# valAug.fit(X_train)


# In[ ]:


early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.00008,
    patience=11,
    verbose=1,
    restore_best_weights=True,
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy',
    min_delta=0.0001,
    factor=0.25,
    patience=4,
    min_lr=1e-7,
    verbose=1,
)

callbacks = [
    early_stopping,
    lr_scheduler,
]


# In[ ]:


batch_size = 32
epochs = 40
opt = Adam(lr=0.001)


# In[ ]:


model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
)

history = model.fit_generator(
    trainAug.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_test, y_test),
    steps_per_epoch=len(X_train) / batch_size,
    epochs=epochs,
    callbacks=callbacks,
    use_multiprocessing=True
)


# In[ ]:


sns.set()
fig = plt.figure(0, (12, 4))

ax = plt.subplot(1, 2, 1)
sns.lineplot(history.epoch, history.history['accuracy'], label='train')
sns.lineplot(history.epoch, history.history['val_accuracy'], label='valid')
plt.title('Accuracy')
plt.tight_layout()

ax = plt.subplot(1, 2, 2)
sns.lineplot(history.epoch, history.history['loss'], label='train')
sns.lineplot(history.epoch, history.history['val_loss'], label='valid')
plt.title('Loss')
plt.tight_layout()

#plt.savefig('epoch_history_resnet7-3-split.png')
plt.show()


# In[ ]:


yhat_test = np.argmax(model.predict(X_test), axis=1)
ytest_ = np.argmax(y_test, axis=1)

scikitplot.metrics.plot_confusion_matrix(ytest_, yhat_test, figsize=(7,7))


# In[ ]:




