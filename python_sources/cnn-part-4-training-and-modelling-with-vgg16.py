#!/usr/bin/env python
# coding: utf-8

# This is the final part of my CNN series for detecting respiratory diseases using respiratory sound. On this kernel, we use the images created from audio to feed to a convolutional 2d model, in this case, I'm using VGG16.
# 
# If you missed out on how we manipulate and transformed out data, you can look through these kernels:
# - Part 1: [Slice audio into subslices based on the txt files](https://www.kaggle.com/danaelisanicolas/cnn-part-1-create-subslices-for-each-sound)
# - Part 2: [Splitting to train and test](https://www.kaggle.com/danaelisanicolas/cnn-part-2-split-to-train-and-test)
# - Part 3: [Convert audio to spectrogram images](https://www.kaggle.com/danaelisanicolas/cnn-part-3-create-spectrogram-images)
# 
# Without further adeu, let's start!

# In[ ]:


get_ipython().system("ls '../input'")


# Import the necessary libraries. Here we're using TensorFlow Keras. Again as mentioned, we'll use VGG16.

# In[ ]:


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import metrics

from sklearn.utils import class_weight
from collections import Counter

import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

import pandas as pd


# In[ ]:


train_loc = '../input/specimages/train_test_split/train/'
test_loc = '../input/specimages/train_test_split/val/'


# Load the train and test data using ImageDataGenerator flow_from_directory. Target size set to 224x224 given that this is the input requirement of VGG16.

# In[ ]:


trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory=train_loc, target_size=(224,224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory=test_loc, target_size=(224,224))


# In[ ]:


diagnosis_csv = '../input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv'
diagnosis = pd.read_csv(diagnosis_csv, names=['pId', 'diagnosis'])
diagnosis.head()


# In[ ]:


categories = diagnosis['diagnosis'].unique()
categories


# Use the weights from imagenet to get the pretrained model. The prediction is set to 8 output because we have 8 classes.

# In[ ]:


vgg16 = VGG16(weights='imagenet')
vgg16.summary()

x  = vgg16.get_layer('fc2').output
prediction = Dense(8, activation='softmax', name='predictions')(x)

model = Model(inputs=vgg16.input, outputs=prediction)


# Now consider that normally, imagenet is recognising objects. We have spectrogram images which has signals. So we set 20 out of 23 layers to trainable.

# In[ ]:


for layer in model.layers:
    layer.trainable = False

for layer in model.layers[-20:]:
    layer.trainable = True
    print("Layer '%s' is trainable" % layer.name)  


# Use Adam as our optimiser and set the learning rate to 0.000001 because again we're trying to recognise signals. Additionally it's not a normal sound wave either cause we're using breathing sounds. So the model has to be very sensitive to small details.

# In[ ]:


opt = Adam(lr=0.000001)
model.compile(optimizer=opt, loss=categorical_crossentropy, 
              metrics=['accuracy', 'mae'])
model.summary()


# Now we also want to save the model if the validation accuracy improved. However, we don't want to waste computational resource if we're seeing the lack of improvement after 20 epochs.
# 
# We then define the checkpoint and early stopping for our model.

# In[ ]:


checkpoint = ModelCheckpoint("vgg16_base_res.h5", monitor='val_accuracy', verbose=1, 
                             save_best_only=True, save_weights_only=False, mode='auto')
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')


# Now, considering our data is imbalanced, we need to compute the class weight for each category/class.

# In[ ]:


counter = Counter(traindata.classes)                       
max_val = float(max(counter.values()))   
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}
class_weights


# Finally everything's set, let's start training!
# 
# Notice that steps_per_epoch and validation_steps were calculated instead of just simply setting a random value. The reason is we need to see the optimal way to getting these values. We calculate this by dividing samples / batch_size.

# In[ ]:


hist = model.fit(traindata, steps_per_epoch=traindata.samples//traindata.batch_size, validation_data=testdata, 
                 class_weight=class_weights, validation_steps=testdata.samples//testdata.batch_size, 
                 epochs=110,callbacks=[checkpoint,early])


# And lastly we want to visualise how our model is learning. Plot the loss, accuracy, and mae as follows:

# In[ ]:


plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='val')
plt.title('VGG16: Loss and Validation Loss (0.000001 = Adam LR)')
plt.legend();
plt.show()

plt.plot(hist.history['accuracy'], label='train')
plt.plot(hist.history['val_accuracy'], label='val')
plt.title('VGG16: Accuracy and Validation Accuracy (0.000001 = Adam LR)')
plt.legend();
plt.show()

plt.plot(hist.history['mae'], label='train')
plt.plot(hist.history['val_mae'], label='val')
plt.title('VGG16: MAE and Validation MAE (0.000001 = Adam LR)')
plt.legend();
plt.show()


# We see that slowly but surely our model is learning our data. Validation accuracy is increasing steadily while mean absolute error is decreasing. However, we should be aware that loss is increasing. In this context, but maybe we can just ditch loss since we are already looking at mae.
# 
# And that's that! If you're following this series, thank you so much! Please upvote if this helped you in any way.
# 
