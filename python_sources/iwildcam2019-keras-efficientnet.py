#!/usr/bin/env python
# coding: utf-8

# **Simple example of transfer learning from pretrained model using Keras and Efficientnet (https://pypi.org/project/efficientnet/).**
# * Metrics: f1_score

# In[ ]:


get_ipython().system('pip install git+https://github.com/qubvel/efficientnet')


# In[ ]:


from efficientnet import EfficientNetB3


# In[ ]:


import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Activation, Dropout, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, applications
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras import backend as K 


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
train_df.head()


# In[ ]:


train_df['category_id'] = train_df['category_id'].astype(str)


# In[ ]:


batch_size=8
img_size = 32
nb_epochs = 5


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)\ntrain_generator = train_datagen.flow_from_dataframe(\n        dataframe = train_df,        \n        directory = '../input/train_images',\n        x_col = 'file_name', y_col = 'category_id',\n        target_size=(img_size,img_size),\n        batch_size=batch_size,\n        class_mode='categorical',\n        subset='training')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "validation_generator  = train_datagen.flow_from_dataframe(\n        dataframe = train_df,        \n        directory = '../input/train_images',\n        x_col = 'file_name', y_col = 'category_id',\n        target_size=(img_size,img_size),\n        batch_size=batch_size,\n        class_mode='categorical',\n        subset='validation')")


# In[ ]:


set(train_generator.class_indices)


# In[ ]:


nb_classes = 14


# In[ ]:


# Metric

def f1_score(y_true, y_pred):
    beta = 1
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=1)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=1)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=1)
    
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    
    return K.mean(((1+beta**2)*precision*recall) / ((beta**2)*precision+recall+K.epsilon()))


# In[ ]:


model_pre = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
model_pre.trainable = False


# In[ ]:


# Freeze some layers
# for layer in model_pre.layers[:-25]:
#     layer.trainable = False


# In[ ]:


#Adding custom layers 
x = model_pre.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(nb_classes, activation="softmax")(x)
model = Model(input = model_pre.input, output = predictions)

model.compile(optimizers.rmsprop(lr=0.001, decay=1e-6),loss='categorical_crossentropy',metrics=[f1_score])


# In[ ]:


model.summary()


# In[ ]:


# Callbacks

checkpoint = ModelCheckpoint("dnet121_1.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Train model\nhistory = model.fit_generator(\n            train_generator,\n#             steps_per_epoch = train_generator.samples // batch_size,\n            steps_per_epoch = 50,\n            validation_data = validation_generator, \n#             validation_steps = validation_generator.samples // batch_size,\n            validation_steps = 25,\n            epochs = nb_epochs,\n            callbacks = [checkpoint, early],\n            verbose=2)')


# In[ ]:


with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['f1_score', 'val_f1_score']].plot()


# ### Prediction

# In[ ]:


test_df = pd.read_csv('../input/test.csv')
test_df.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "test_datagen = ImageDataGenerator(rescale=1./255)\ntest_generator = test_datagen.flow_from_dataframe(\n        dataframe = test_df,        \n        directory = '../input/test_images',\n        x_col = 'file_name', y_col = None,\n        target_size = (img_size,img_size),\n        batch_size = 1,\n        shuffle = False,\n        class_mode = None\n        )")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'test_generator.reset()\npredict = model.predict_generator(test_generator, steps = len(test_generator.filenames))')


# In[ ]:


len(predict)


# In[ ]:


predicted_class_indices=np.argmax(predict,axis=1)


# In[ ]:


labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


# In[ ]:


sam_sub_df = pd.read_csv('../input/sample_submission.csv')
print(sam_sub_df.shape)
sam_sub_df.head()


# In[ ]:


filenames=test_generator.filenames
results=pd.DataFrame({"Id":filenames,
                      "Predicted":predictions})
results['Id'] = results['Id'].map(lambda x: str(x)[:-4])
results.to_csv("results.csv",index=False)


# In[ ]:


results.head()

