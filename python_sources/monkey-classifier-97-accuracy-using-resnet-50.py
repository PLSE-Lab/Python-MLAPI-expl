#!/usr/bin/env python
# coding: utf-8

# hello kagglers, this is my first kernel on kaggle. I would like to show my Monkey Classifier using ResNet-50. I built it using Keras pre trained model, with difference on last few layer. Feel free to write any advice for on the comment section, thanks!

# **1. Importing required Library**

# In[11]:


from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator, image
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGE = True


# **2. Define some constant needed throughout the script**

# In[12]:


N_CLASSES = 10
EPOCHS = 15
PATIENCE = 5
TRAIN_PATH= '../input/training/training/'
VALID_PATH = '../input/validation/validation/'
MODEL_CHECK_WEIGHT_NAME = 'resnet_monki_v1_chk.h5'


# **3. Define model to be used**
# we freeze the pre trained resnet model weight, and add few layer on top of it to utilize our custom dataset
# 

# In[13]:


K.set_learning_phase(0)
model = ResNet50(input_shape=(224,224,3),include_top=False, weights='imagenet', pooling='avg')
K.set_learning_phase(1)
x = model.output
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(N_CLASSES, activation='softmax', name='custom_output')(x)
custom_resnet = Model(inputs=model.input, outputs = output)

for layer in model.layers:
    layer.trainable = False

custom_resnet.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
custom_resnet.summary()


# **4. Load dataset to be used**

# In[14]:


datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
traingen = datagen.flow_from_directory(TRAIN_PATH, target_size=(224,224), batch_size=32, class_mode='categorical')
validgen = datagen.flow_from_directory(VALID_PATH, target_size=(224,224), batch_size=32, class_mode='categorical', shuffle=False)


# **5. Train Model**
# we use ModelCheckpoint to save the best model based on validation accuracy
# 

# In[15]:


es_callback = EarlyStopping(monitor='val_acc', patience=PATIENCE, mode='max')
mc_callback = ModelCheckpoint(filepath=MODEL_CHECK_WEIGHT_NAME, monitor='val_acc', save_best_only=True, mode='max')
train_history = custom_resnet.fit_generator(traingen, steps_per_epoch=len(traingen), epochs= EPOCHS, validation_data=traingen, validation_steps=len(validgen), verbose=2, callbacks=[es_callback, mc_callback])


# In[16]:


plt.figure(1)
plt.subplot(221)
plt.plot(train_history.history['acc'])
plt.plot(train_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train accuracy','validation accuracy'])

plt.subplot(222)
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train loss', 'validation loss'])

plt.show()


# **7. Load Last Checkpoint Weight**

# In[17]:


custom_resnet.load_weights(MODEL_CHECK_WEIGHT_NAME)


# **7. Print validation confusion matrix, classification report, and accuracy**

# In[18]:


predict = custom_resnet.predict_generator(validgen, steps=len(validgen), verbose=1)
test_labels = validgen.classes
confusion_matrix(test_labels, predict.argmax(axis=1))


# In[19]:


cr_labels = list(validgen.class_indices.keys())
classification_report(test_labels, predict.argmax(axis=1), target_names=cr_labels)


# In[20]:


accuracy_score(test_labels,predict.argmax(axis=1))

