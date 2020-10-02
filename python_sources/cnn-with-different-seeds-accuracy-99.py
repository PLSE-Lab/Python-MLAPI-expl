#!/usr/bin/env python
# coding: utf-8

# ## Summary
# We have used data augmentation and a CNN model implemented with Keras framework. We have trained the same model (same architectura, same hyperparameters and data) with different seed inicializations. The accuracy has been very different depending of the seed used. In the best case we have been able to get an accuracy bigger than 99%. 

# ## Data loading and quick analysis

# In[1]:


import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter("ignore", category=FutureWarning)


# In[2]:


data_dir = "../input/Sign-language-digits-dataset/"


# In[3]:


X = np.load(data_dir + "X.npy")
y = np.load(data_dir + "Y.npy")


# In[4]:


plt.figure(figsize=(10,10))
for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.tight_layout()
    plt.imshow(X[i], cmap=plt.get_cmap('gray'))
    plt.axis('off')
    plt.title(y[i].argmax())    
plt.show()


# In[5]:


X.shape


# In[6]:


X.min(), X.max()


# We have aprox the same number of examples per each class: 

# In[7]:


num_cases = np.unique(y.argmax(axis=1), return_counts=True) 


# In[8]:


plt.title("Number of cases")
plt.xticks(num_cases[0])
plt.bar(num_cases[0], num_cases[1])
plt.show()


# We are going to fix this seed to have reproducibility (the data will be splited into training and validation sets always in the same way):

# In[9]:


import tensorflow as tf
import random as rn
from keras import backend as K

os.environ['PYTHONHASHSEED'] = '0'

SEED = 1
np.random.seed(SEED)
rn.seed(SEED)


# In[10]:


img_rows , img_cols, img_channel = 64, 64, 1
target_size = (img_rows, img_cols)
target_dims = (img_rows, img_cols, img_channel) 
n_classes = 10
val_frac = 0.2
batch_size = 64


# In[11]:


X = X.reshape(X.shape[0], img_rows, img_cols, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_frac, stratify=y, random_state=SEED)


# ## Model

# In[12]:


from keras.layers import Dropout, Flatten, Dense, Convolution2D, MaxPooling2D, Input, BatchNormalization
from keras.models import Sequential, Model 
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from tensorflow import set_random_seed


# In[13]:


def initialize_nn_seed(seed):
    np.random.seed(seed)
    rn.seed(seed)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(seed)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)    


# In[14]:


def create_model(seed):
    initialize_nn_seed(seed)    
    
    model = Sequential()
    model.add(Convolution2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(64, 64, 1)))
    model.add(Convolution2D(16, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(Convolution2D(32, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, kernel_size=(2, 2), strides=(1, 1), activation='relu'))
    model.add(Convolution2D(64, kernel_size=(2, 2), strides=(2, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-3), metrics=["accuracy"])
    return model


# In[15]:


def train_model(model): 
    epochs = 40 
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    checkpoint = ModelCheckpoint("weights-best.hdf5", monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    annealer = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=10*batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, checkpoint, annealer]
    )  
    model.load_weights("weights-best.hdf5")
    return history, model


# To get augmentated data:

# In[16]:


datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range = 0.1,
                             fill_mode="nearest")
        
train_generator = datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True, seed=SEED, subset="training")


# The process of training a neural network using a gpu is non deterministic (at least with Tensorflow as backend at the moment) but fixing a seed the results can be quite similar as can be seen in the kernel [Keras GPU/CPU Reproducibility Test
# ](https://www.kaggle.com/lbronchal/keras-gpu-cpu-reproducibility-test). We are going to fix a seed for the training process. We have seen that for this particular neural network the seed is very important:

# ### Training with SEED = 123456
# If we initialize with this particular seed, the initial random weights and the rest of configuration are not going to find a good minimum: the accuracy of the model is very bad.

# In[17]:


model = create_model(seed=123456)
history, model = train_model(model)


# In[18]:


scores = model.evaluate(X_test, y_test, verbose=0)
print("{}: {:.2f}%".format(model.metrics_names[1], scores[1]*100))


# ### Training with SEED = 1
# With this other different inicialization the training process is able to find a better minimum and the model achieve a good accuracy

# In[19]:


model = create_model(seed=1)
history, model = train_model(model)


# In[20]:


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[21]:


scores = model.evaluate(X_test, y_test, verbose=0)
print("{}: {:.2f}%".format(model.metrics_names[1], scores[1]*100))


# Let's see the wrong cases this model is not able to identify:

# In[22]:


predicted_classes = model.predict_classes(X_test)
y_true = y_test.argmax(axis=1)


# In[23]:


import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y_true, predicted_classes, normalize=True, figsize=(8, 8))
plt.show()


# In[24]:


correct = np.where(predicted_classes == y_true)[0]
incorrect = np.where(predicted_classes != y_true)[0]


# In[25]:


plt.figure(figsize=(8, 8))
for i, correct in enumerate(incorrect[:9]):
    plt.subplot(430+1+i)
    plt.imshow(X_test[correct].reshape(img_rows, img_cols), cmap='gray')
    plt.title("Pred: {} || Class {}".format(predicted_classes[correct], y_true[correct]))
    plt.axis('off')
    plt.tight_layout()
plt.show()


# In[26]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, predicted_classes))


# ## Conclusion
# We have trained the same model with exactly the same parameters and data but with diferent seed inicialization. In one of the cases the performance was very bad (the training process wasn't able to scape from a local minimum) but, in the other, the model was able to achieve a quit good accuracy bigger than 99%. When building a deep learning model it can be interesting to try different seeds as the result can be very different. This affect only to the training process: once the weights has been obtained the prediction for new cases is a deterministic process.
# 
