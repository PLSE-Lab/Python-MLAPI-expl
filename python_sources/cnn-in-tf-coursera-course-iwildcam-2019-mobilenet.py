#!/usr/bin/env python
# coding: utf-8

# Continuation of my learning through Convolutional Neural Networks in TensorFlow Coursera course 
# 
# https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/
# 
# https://www.kaggle.com/rblcoder/learning-cnn-in-tensorflow-coursera-course

# 14 classes

# In[ ]:


import os

print(os.listdir("../input"))


# In[ ]:


print(os.listdir("../input/cnn-in-tf-coursera-course-iwildcam-2019-mobilenet/"))


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import os
IMAGE_HT_WID = 64
print(os.listdir("../input/iwildcam-2019-fgvc6/"))
df_train = pd.read_csv("../input/iwildcam-2019-fgvc6/train.csv")
df_train.head()


# In[ ]:


df_train.category_id.value_counts()


# In[ ]:


df_train.category_id.nunique()


# In[ ]:


df_train.info()


# In[ ]:


df_train['category_id'] = df_train['category_id'].astype(str)


# In[ ]:


n_classes = 14


# In[ ]:


BATCH_SIZE = 120


# In[ ]:


#https://stackoverflow.com/questions/44114463/stratified-sampling-in-pandas
#df_train_temp = df_train.groupby('category_id', group_keys=False).apply(lambda x: x.sample(min(len(x), 200)))


# In[ ]:


#df_train_temp['category_id'].value_counts()


# In[ ]:


#https://keras.io/preprocessing/image/
#https://medium.com/@arindambaidya168/https-medium-com-arindambaidya168-using-keras-imagedatagenerator-b94a87cdefad
#https://github.com/lmoroney/dlaicourse/blob/master/Exercises/Exercise%205%20-%20Real%20World%20Scenarios/Exercise%205%20-%20Answer.ipynb
#https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c
# import tensorflow as tf
# from keras_preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import RMSprop
#IMAGE_HT_WID = 96

#datagen=ImageDataGenerator(rescale=1./255, validation_split=0.2)

# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True, 
#         validation_split=0.1)

train_datagen = ImageDataGenerator(
#                                rotation_range=15,
                               width_shift_range=0.1,
#                                height_shift_range=0.1,
#                                shear_range=0.01,
#                                zoom_range=[0.9, 1.25],
#                                horizontal_flip=True,
#                                vertical_flip=False,
#                                fill_mode='reflect',
#                                #data_format='channels_last',
#                                brightness_range=[0.5, 1.5],
                               validation_split=0.2,
                               rescale=1./255)


test_datagen = ImageDataGenerator(rescale=1./255)
#test_datagen = ImageDataGenerator()

train_generator=train_datagen.flow_from_dataframe(
                    dataframe=df_train,
                    directory="../input/iwildcam-2019-fgvc6/train_images/",
                    x_col="file_name",
                    y_col="category_id",
                    subset="training",
                    batch_size=BATCH_SIZE,
                    seed=424,
                    shuffle=True,
                    class_mode="categorical",
                    color_mode = 'grayscale',
                    target_size=(IMAGE_HT_WID,IMAGE_HT_WID))

valid_generator=train_datagen.flow_from_dataframe(
                    dataframe=df_train,
                    directory="../input/iwildcam-2019-fgvc6/train_images/",
                    x_col="file_name",
                    y_col="category_id",
                    subset="validation",
                    batch_size=BATCH_SIZE,
                    seed=424,
                    shuffle=True,
                    class_mode="categorical",
                    color_mode = 'grayscale',
                    target_size=(IMAGE_HT_WID,IMAGE_HT_WID))

from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np
#n_classes = 14

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMAGE_HT_WID, IMAGE_HT_WID, 3)),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dropout(0.8),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(14, activation='softmax')
# ])


# https://www.kaggle.com/gaborfodor/greyscale-mobilenet-lb-0-892

# In[ ]:


# def top_3_accuracy(y_true, y_pred):
#     return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


# In[ ]:


IMG_SHAPE = (IMAGE_HT_WID, IMAGE_HT_WID, 1)
#https://www.kaggle.com/ratthachat/fat19-keras-baseline-on-preprocesseddata-lb576
# Create the base model from the pre-trained model MobileNet V2
model = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE,
                                              # include_top=False, 
                                              # weights='imagenet'
                                                alpha=1.,
                                                classes = n_classes,
                                                 weights=None,
                                                 pooling = 'avg'
                                                 )

# base_model = tf.keras.applications.vgg16.VGG16(input_shape=IMG_SHAPE,
#                                                include_top=False, 
#                                                weights='imagenet')

# base_model.trainable = False
# model = tf.keras.Sequential([
#   base_model,
#   tf.keras.layers.GlobalAveragePooling2D(),
#   #tf.keras.layers.BatchNormalization(),    
#   #tf.keras.layers.Dropout(0.5),  
#   tf.keras.layers.Dense(256,activation='relu'),  
#   #tf.keras.layers.BatchNormalization(),    
#   #tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(n_classes, activation='softmax')
  
# ])

# #model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
# model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.2), loss='categorical_crossentropy',
              metrics=[tf.keras.metrics.categorical_crossentropy, tf.keras.metrics.categorical_accuracy])
#https://stackoverflow.com/questions/41859997/keras-model-load-weights-for-neural-net
model.load_weights('../input/cnn-in-tf-coursera-course-iwildcam-2019-mobilenet/weights-improvement.06-0.90.hdf5')


# In[ ]:


model.summary()


# In[ ]:


# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                               patience=5, min_lr=0.001)
# redlrOnPlateau = keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.75, patience=3, min_delta=0.001,
#                           mode='max', min_lr=1e-5, verbose=1),



# In[ ]:


#https://machinelearningmastery.com/check-point-deep-learning-models-keras/
#https://keras.io/callbacks/#modelcheckpoint
checkpoint = keras.callbacks.ModelCheckpoint('weights-improvement.{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5', monitor='val_categorical_accuracy', 
                                                verbose=1, save_best_only=True, save_weights_only=True, mode='max', period=1)


# In[ ]:


##https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c
EPOCHS=4
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size + 1
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size + 1

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.7,
                              patience=5, min_lr=1e-5)
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=EPOCHS,
                    callbacks=[reduce_lr,checkpoint],          
                    workers=4,
                    verbose=2
)


# In[ ]:


#https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c
# EPOCHS=16
# STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size + 1
# STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size + 1
# history = model.fit_generator(generator=train_generator,
#                     steps_per_epoch=STEP_SIZE_TRAIN,
#                     validation_data=valid_generator,
#                     validation_steps=STEP_SIZE_VALID,
#                     epochs=EPOCHS,
#                     callbacks=[checkpoint, redlrOnPlateau],          
#                     workers=4,
#                     verbose=2
# )


# In[ ]:


import matplotlib.pyplot as plt
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training categorical_accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation categorical_accuracy')
plt.title('Training and validation categorical_accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()


# In[ ]:


import matplotlib.pyplot as plt
acc = history.history['categorical_crossentropy']
val_acc = history.history['val_categorical_crossentropy']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training categorical_crossentropy')
plt.plot(epochs, val_acc, 'b', label='Validation categorical_crossentropy')
plt.title('Training and validation categorical_crossentropy')
plt.legend(loc=0)
plt.figure()


plt.show()


# In[ ]:


df_test = pd.read_csv("../input/iwildcam-2019-fgvc6/test.csv")
df_test.head()


# In[ ]:


#https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c
#test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
                dataframe=df_test,
                directory="../input/iwildcam-2019-fgvc6/test_images/",
                x_col="file_name",
                y_col=None,
                batch_size=BATCH_SIZE,
                seed=42,
                shuffle=False,
                class_mode=None,
                color_mode = 'grayscale',
                target_size=(IMAGE_HT_WID,IMAGE_HT_WID))
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size + 1
test_generator.reset()
pred=model.predict_generator(test_generator,
                steps=STEP_SIZE_TEST,
                verbose=1)


# In[ ]:


#https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c
predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


# In[ ]:


submission=pd.DataFrame({"Id":df_test.id,
                      "Predicted":predictions})
submission.to_csv("submission.csv",index=False)


# In[ ]:


#https://www.tensorflow.org/tutorials/images/transfer_learning
# acc = history.history['acc']
# val_acc = history.history['val_acc']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.ylabel('Accuracy')
# plt.ylim([min(plt.ylim()),1])
# plt.title('Training and Validation Accuracy')

# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.ylabel('Cross Entropy')
# plt.ylim([0,max(plt.ylim())])
# plt.title('Training and Validation Loss')
# plt.show()

