#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(style="darkgrid")

import warnings
warnings.filterwarnings("ignore")

import gc


# In[ ]:


import tensorflow as tf
from tensorflow.keras import applications 
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import backend as k 
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# # Image Generator

# In[ ]:


SIZE = 224
NUM_CLASSES = 5
BATCH_SIZE = 32


# ## Enhance Image

# In[ ]:


import cv2

def grayCLAHE(img):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY).astype(np.uint8)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img2 = clahe.apply(img)
    
    claheImg = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
    
    return claheImg


# In[ ]:


def preprocessImage(img):
    #img = grayCLAHE(img)    
    return applications.mobilenet_v2.preprocess_input(img)


# In[ ]:


img = cv2.imread('../input/aptos2019-blindness-detection/train_images/ae49cc60f251.png',-1)
print(img.shape)
_, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
axes[0].imshow(img);
axes[1].imshow(preprocessImage(img)); 


# ## Train Data Generator

# In[ ]:


train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
train_df['FILE'] = [f"{x.id_code}.png" for _,x in train_df.iterrows()]
train_df.diagnosis = train_df.diagnosis.astype('str')
train_df.head()


# In[ ]:


image_datagen = ImageDataGenerator(validation_split=0.1, 
                                   #rescale=1./255.,
                                   #featurewise_center=True,
                                   #samplewise_center=False,
                                   #featurewise_std_normalization=True,
                                   #samplewise_std_normalization=False,
                                   rotation_range=360,
                                   #shear_range=5,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True, 
                                   vertical_flip=True,
                                   preprocessing_function=preprocessImage #applications.mobilenet_v2.preprocess_input
                                  )

TRAIN_IMAGE_DIRECTORY = '../input/aptos2019-blindness-detection/train_images/'

train_generator = image_datagen.flow_from_dataframe(train_df, 
                                                    directory=TRAIN_IMAGE_DIRECTORY, 
                                                    subset='training', 
                                                    shuffle=True,
                                                    drop_duplicates=False, 
                                                    color_mode='rgb',
                                                    x_col='FILE', 
                                                    #y_col=['D_0','D_1','D_2','D_3','D_4'],
                                                    #class_mode='other', 
                                                    y_col='diagnosis',
                                                    class_mode='categorical',
                                                    batch_size=BATCH_SIZE, 
                                                    target_size=(SIZE,SIZE), 
                                                    #seed=42
                                                   )

valid_generator = image_datagen.flow_from_dataframe(train_df, 
                                                    directory=TRAIN_IMAGE_DIRECTORY, 
                                                    subset='validation', 
                                                    shuffle=False,
                                                    drop_duplicates=False, 
                                                    color_mode='rgb',
                                                    x_col='FILE', 
                                                    #y_col=['D_0','D_1','D_2','D_3','D_4'],
                                                    #class_mode='other', 
                                                    y_col='diagnosis',
                                                    class_mode='categorical',
                                                    batch_size=BATCH_SIZE, 
                                                    target_size=(SIZE,SIZE),
                                                    #seed=42
                                                   )


# In[ ]:


NUM_TRAIN_IMAGES = len(train_generator.filenames)
NUM_VALID_IMAGES = len(valid_generator.filenames)


# In[ ]:


train_generator.class_indices


# # Test Data Generator

# In[ ]:


test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
test_df['FILE'] = [f"{x.id_code}.png" for _,x in test_df.iterrows()]
test_df.head()


# In[ ]:


test_datagen = ImageDataGenerator(#rescale=1./255.,
                                  #featurewise_center=True,
                                  #samplewise_center=False,
                                  #featurewise_std_normalization=True,
                                  #samplewise_std_normalization=False,
                                  preprocessing_function=preprocessImage #applications.mobilenet_v2.preprocess_input
                                 )

test_generator = test_datagen.flow_from_dataframe(test_df, 
                                                  directory='../input/aptos2019-blindness-detection/test_images/', 
                                                  shuffle=False,
                                                  drop_duplicates=False, 
                                                  color_mode='rgb',
                                                  x_col='FILE', y_col=None,
                                                  batch_size=2, 
                                                  target_size=(SIZE,SIZE), 
                                                  class_mode=None
                                                 )


# # MobileNetV2 
# ## Download Trained Model

# In[ ]:


mnet2 = applications.mobilenet_v2.MobileNetV2(weights = None, include_top=True, input_shape = (SIZE, SIZE, 3) )
mnet2.load_weights('../input/mobilenetv2-full-weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5')

mnet2.summary()


# ## Customize MobileNetV2 Head

# In[ ]:


x = mnet2.layers[-2].output
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512)(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

custom_model = Model(inputs=mnet2.input, outputs=predictions)

for layer in custom_model.layers[:-10]:
    layer.trainable = False

for i, layer in enumerate(custom_model.layers):
    print(f'layer {i}: `{layer.name}` {layer.trainable}')
#custom_model.summary()


# In[ ]:


# from tensorflow.keras.optimizers import *

custom_model.compile(loss='categorical_crossentropy',
                     #optimizer=Adam(0.0001),
                     optimizer=SGD(lr=1.62e-2, decay=1e-6, momentum=0.9, nesterov=True),
                     #optimizer=Adagrad(0.0001),
                     metrics=['accuracy'])

gc.collect()


# # Train Model

# In[ ]:


class expandTrainableLayers(tf.keras.callbacks.Callback):
    def reportLayers(self):
        for i, layer in enumerate(custom_model.layers):
                print(f'layer {i}: `{layer.name}` {layer.trainable}')
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch+1==1:
            for layer in custom_model.layers:
                layer.trainable = True
                
            for layer in custom_model.layers[:-50]:
                layer.trainable = False
            
            self.reportLayers()
            
        if epoch+1==2:
            for layer in custom_model.layers:
                layer.trainable = True
                
            for layer in custom_model.layers[:-100]:
                layer.trainable = False
            
            self.reportLayers()

        if epoch+1==3:
            for layer in custom_model.layers:
                layer.trainable = True

            self.reportLayers()
            
        gc.collect()


# In[ ]:


#from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=.5, patience=3, verbose=1, 
                              mode='max', min_delta=0.001, cooldown=0, min_lr=0)

check_pt = ModelCheckpoint('../checkpoint', monitor='val_acc', verbose=1, save_best_only=True, 
                           save_weights_only=False, mode='max', save_freq='epoch')

early_stop = EarlyStopping(monitor='val_acc', patience=10, verbose=1, 
                           mode='max', baseline=None, restore_best_weights=True)

moreTrainableLayers = expandTrainableLayers()

gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = custom_model.fit_generator(train_generator,\n                                     steps_per_epoch=2*NUM_TRAIN_IMAGES // BATCH_SIZE,\n                                     validation_data=valid_generator,\n                                     validation_steps= NUM_VALID_IMAGES // BATCH_SIZE,\n                                     callbacks=[lr_reduce, check_pt, early_stop, moreTrainableLayers],\n                                     epochs=70)\ngc.collect()')


# In[ ]:


for metric in custom_model.metrics_names:
    # Plot training & validation loss and metrics
    ROLLING_WINDOW = 5
    
    plt.figure(figsize=(16,7))
    plt.plot(history.history[metric])
    plt.plot(pd.DataFrame(data=history.history[metric]).             rename(columns={0:f'{metric} {ROLLING_WINDOW} Rolling Average'}).             rolling(ROLLING_WINDOW).mean())
    plt.plot(history.history[f'val_{metric}'])
    plt.plot(pd.DataFrame(data=history.history[f'val_{metric}']).             rename(columns={0:f'val_{metric} {ROLLING_WINDOW} Rolling Average'}).             rolling(ROLLING_WINDOW).mean())
    
    plt.title(f'Model {metric}')
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right');
    
    plt.show()
    
    gc.collect()


# In[ ]:


from sklearn.metrics import classification_report

report = classification_report(y_true=valid_generator.classes, 
                               y_pred=np.argmax(custom_model.predict_generator(valid_generator),axis=1),
                               target_names=list(valid_generator.class_indices.keys())
                              )

print(report)


# # Prediction

# In[ ]:


get_ipython().run_cell_magic('time', '', 'pred = custom_model.predict_generator(test_generator, steps=test_df.shape[0]//2)\ngc.collect()')


# In[ ]:


PREDICTION = np.argmax(pred, axis=1) 
PREDICTION.shape


# In[ ]:


test_df['diagnosis'] = PREDICTION
test_df[['id_code','diagnosis']]


# In[ ]:


test_df[['id_code','diagnosis']].to_csv('submission.csv',index=False)


# In[ ]:




