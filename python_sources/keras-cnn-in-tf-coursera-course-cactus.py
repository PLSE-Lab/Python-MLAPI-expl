#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip uninstall tensorflow -y')


# In[ ]:


#https://www.kaggle.com/vladminzatu/cactus-detection-with-tensorflow-2-0
#!pip install tensorflow==2.0.0-alpha0
#https://medium.com/tensorflow/test-drive-tensorflow-2-0-alpha-b6dd1e522b01
get_ipython().system('pip install tensorflow-gpu==2.0.0-alpha0')
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True


# Continuation of applying the learning from Convolutional Neural Networks in TensorFlow Coursera course 
# 
# https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/
# 
# https://www.kaggle.com/rblcoder/learning-cnn-in-tensorflow-coursera-course

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
batch_size = 120


# In[ ]:


df_train = pd.read_csv('../input/train.csv')


# In[ ]:


df_test = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


df_train.info()


# In[ ]:


df_train.has_cactus.value_counts()


# In[ ]:


df_train['has_cactus'] = df_train['has_cactus'].astype(str)


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# https://medium.com/@arindambaidya168/https-medium-com-arindambaidya168-using-keras-imagedatagenerator-b94a87cdefad
# 
# https://github.com/lmoroney/dlaicourse/blob/master/Exercises/Exercise%205%20-%20Real%20World%20Scenarios/Exercise%205%20-%20Answer.ipynb

# In[ ]:


#https://keras.io/preprocessing/image/
#https://medium.com/@arindambaidya168/https-medium-com-arindambaidya168-using-keras-imagedatagenerator-b94a87cdefad
#https://github.com/lmoroney/dlaicourse/blob/master/Exercises/Exercise%205%20-%20Real%20World%20Scenarios/Exercise%205%20-%20Answer.ipynb
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
#datagen=ImageDataGenerator(rescale=1./255, validation_split=0.2)
IMAGE_HT_WID = 75

#https://stackoverflow.com/questions/50133385/preprocessing-images-generated-using-keras-function-imagedatagenerator-to-trai
train_datagen = ImageDataGenerator(
                                rotation_range=15,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.01,
                               zoom_range=[0.9, 1.25],
                               horizontal_flip=True,
                               vertical_flip=False,
                               #data_format='channels_last',
                              fill_mode='reflect',
                              brightness_range=[0.5, 1.5],
                               channel_shift_range=30,
                               validation_split=0.2,
                              # preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input,
                               rescale=1./255)


test_datagen = ImageDataGenerator(
    #preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input,
                                  rescale=1./255)
#test_datagen = ImageDataGenerator()

train_generator=train_datagen.flow_from_dataframe(
                    dataframe=df_train,
                    directory="../input/train/train/",
                    x_col="id",
                    y_col="has_cactus",
                    subset="training",
                    batch_size=batch_size,
                    seed=42,
                    shuffle=True,
                    class_mode="categorical",
                    target_size=(IMAGE_HT_WID,IMAGE_HT_WID))

valid_generator=train_datagen.flow_from_dataframe(
                    dataframe=df_train,
                    directory="../input/train/train/",
                    x_col="id",
                    y_col="has_cactus",
                    subset="validation",
                    batch_size=batch_size,
                    seed=42,
                    shuffle=True,
                    class_mode="categorical",
                    target_size=(IMAGE_HT_WID,IMAGE_HT_WID))

from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np


# In[ ]:


#https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
#from keras.callbacks import Callback
# from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
# class Metrics(tf.keras.callbacks.Callback):
    
#     def on_train_begin(self, logs={}):
#         self.val_f1s = []
#         self.val_recalls = []
#         self.val_precisions = []


#     def on_epoch_end(self, epoch, logs={}):
#         #print(self.model)
# #         val_predict = (np.asarray(self.model.predict_(self.model.validation_data[0]))).round()
# #         val_targ = self.model.validation_data[1]
# #         #val_predict = self.model.valid_generator
# #         _val_f1 = f1_score(val_targ, val_predict)
# #         _val_recall = recall_score(val_targ, val_predict)
# #         _val_precision = precision_score(val_targ, val_predict)
# #         self.val_f1s.append(_val_f1)
# #         self.val_recalls.append(_val_recall)
# #         self.val_precisions.append(_val_precision)
# #         print(_val_f1)
# #         print(_val_precision)
# #         print(_val_recall)
#         return
 

# metrics = Metrics()


# In[ ]:


#https://www.tensorflow.org/alpha/guide/keras/training_and_evaluation
class CatgoricalTruePositives(tf.keras.metrics.Metric):

    def __init__(self, name='binary_true_positives', **kwargs):
      super(CatgoricalTruePositives, self).__init__(name=name, **kwargs)
      self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
      y_pred = tf.argmax(y_pred)
      values = tf.equal(tf.cast(y_true, 'int32'), tf.cast(y_pred, 'int32'))
      values = tf.cast(values, 'float32')
      if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, 'float32')
        values = tf.multiply(values, sample_weight)
      return self.true_positives.assign_add(tf.reduce_sum(values))  # TODO: fix

    def result(self):
      return tf.identity(self.true_positives)  # TODO: fix

    def reset_states(self):
      # The state of the metric will be reset at the start of each epoch.
      self.true_positives.assign(0.)


# In[ ]:


#https://www.tensorflow.org/tutorials/images/transfer_learning
#https://keras.io/applications/#fine-tune-inceptionv3-on-a-new-set-of-classes
import tensorflow as tf
from tensorflow import keras
base_model = tf.keras.applications.ResNet50(input_shape=(IMAGE_HT_WID, IMAGE_HT_WID, 3),
                                               include_top=False, 
                                               weights='imagenet'
                                               )
base_model.trainable = False

#print(base_model.summary())
model = tf.keras.Sequential([
  base_model,
  keras.layers.GlobalAveragePooling2D(),
  #keras.layers.Dense(512, activation='relu'),
  keras.layers.Dropout(.3),
  keras.layers.Flatten(),  
  keras.layers.Dense(2, activation='sigmoid')
])


model.compile(optimizer=tf.keras.optimizers.SGD(lr=.01, momentum=.9), 
              loss='categorical_crossentropy', 
              metrics=[CatgoricalTruePositives()])



#print(model.summary())


# In[ ]:


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, 
                                                                    patience=3, verbose=2, mode='auto',
                                                                    min_lr=1e-6)


# In[ ]:


EPOCHS=12
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size + 1
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size + 1
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=EPOCHS,
                    callbacks=[reduce_lr],          
                    verbose=1
)


# In[ ]:


for i, layer in enumerate(base_model.layers):
   print(i, layer.name)


# In[ ]:


import matplotlib.pyplot as plt
acc = history.history['binary_true_positives']
val_acc = history.history['val_binary_true_positives']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training binary_true_positives')
plt.plot(epochs, val_acc, 'b', label='Validation binary_true_positives')
plt.title('Training and validation binary_true_positives')
plt.legend(loc=0)
plt.figure()


plt.show()


# In[ ]:


STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size + 1
valid_generator.reset()
pred_v=model.predict_generator(valid_generator,
                steps=STEP_SIZE_VALID,
                verbose=1)


# In[ ]:


predicted_v_class_indices=np.argmax(pred_v,axis=1)


# In[ ]:


np.histogram(pred_v)


# In[ ]:


pred_v[:5]


# In[ ]:


predicted_v_class_indices[:5]


# In[ ]:


valid_generator.classes[:5]


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(valid_generator.classes, predicted_v_class_indices))


# In[ ]:


#test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
                dataframe=df_test,
                directory="../input/test/test/",
                x_col="id",
                y_col=None,
                batch_size=batch_size,
                seed=42,
                shuffle=False,
                class_mode=None,
                target_size=(IMAGE_HT_WID,IMAGE_HT_WID))
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size + 1
test_generator.reset()
pred=model.predict_generator(test_generator,
                steps=STEP_SIZE_TEST,
                verbose=1)


# In[ ]:


predicted_class_indices=np.argmax(pred,axis=1)


# In[ ]:


#id = [*range(1,len(df_test)+1)]
#dataframe_output=pd.DataFrame({"id":id})
#dataframe_output["has_cactus"]=predicted_class_indices
#dataframe_output.to_csv("submission.csv",index=False)
df_test["has_cactus"]=predicted_class_indices
df_test.to_csv("submission.csv",index=False)


# In[ ]:


df_test.head()


# In[ ]:


df_test.has_cactus.value_counts()


# In[ ]:


#https://www.tensorflow.org/tutorials/keras/save_and_restore_models
model.save_weights('Cactus_checkpoint1')

