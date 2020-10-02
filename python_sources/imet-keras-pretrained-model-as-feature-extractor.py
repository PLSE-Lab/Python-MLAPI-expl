#!/usr/bin/env python
# coding: utf-8

# ![](https://raw.githubusercontent.com/visipedia/imet-fgvcx/master/assets/banner.png)
# <h1><center>Using the bottleneck features of a pre-trained network</center></h1>
# 
# #### I'm sharing this code since it can get a little tricky to beginners do this, especially finding the right methods, APIs and models, I hope this can help someone.
# 
# #### Using the bottleneck features of a pre-trained model is basically using the models as a preprocessing step on your data, so you get a model (VGG16 in this case) and pass your data through it, the output will be the representation of your data according to the model. Then take these features and use on any other model (another deep learning model or even SVM).
# 
# #### What you need to know:
# - Similar to fine-tuning you need to remove the top of the pre-trained model.
# - Essentially it works as a pipeline with two models.
# - To have good results with this approach your data need to be similar to the same data used to train the model (ImageNet).
# - This will make your training a lot faster since you only need to pass all data once through the big model (VGG 16) than just train a smaller one on a less complex data.

# ### Dependencies

# In[1]:


import os
import cv2
import math
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, fbeta_score
from keras import optimizers
from keras import backend as K
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation, BatchNormalization

# Set seeds to make the experiment more reproducible.
from tensorflow import set_random_seed
from numpy.random import seed
set_random_seed(0)
seed(0)

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid")
warnings.filterwarnings("ignore")


# ### Load data

# In[2]:


train = pd.read_csv('../input/imet-2019-fgvc6/train.csv')
labels = pd.read_csv('../input/imet-2019-fgvc6/labels.csv')
test = pd.read_csv('../input/imet-2019-fgvc6/sample_submission.csv')

train["attribute_ids"] = train["attribute_ids"].apply(lambda x:x.split(" "))
train["id"] = train["id"].apply(lambda x: x + ".png")
test["id"] = test["id"].apply(lambda x: x + ".png")

print('Number of train samples: ', train.shape[0])
print('Number of test samples: ', test.shape[0])
print('Number of labels: ', labels.shape[0])
display(train.head())
display(labels.head())


# In[3]:


# Parameters
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.0001
HEIGHT = 128
WIDTH = 128
CANAL = 3
N_CLASSES = labels.shape[0]
ES_PATIENCE = 5
DECAY_DROP = 0.5
DECAY_EPOCHS = 10
classes = list(map(str, range(N_CLASSES)))


# In[4]:


def f2_score_thr(threshold=0.5):
    def f2_score(y_true, y_pred):
        beta = 2
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold), K.floatx())

        true_positives = K.sum(K.clip(y_true * y_pred, 0, 1), axis=1)
        predicted_positives = K.sum(K.clip(y_pred, 0, 1), axis=1)
        possible_positives = K.sum(K.clip(y_true, 0, 1), axis=1)

        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())

        return K.mean(((1+beta**2)*precision*recall) / ((beta**2)*precision+recall+K.epsilon()))
    return f2_score

def step_decay(epoch):
    initial_lrate = LEARNING_RATE
    drop = DECAY_DROP
    epochs_drop = DECAY_EPOCHS
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    
    return lrate


# In[5]:


train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train,
    directory="../input/imet-2019-fgvc6/train",
    x_col="id",
    y_col="attribute_ids",
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode=None,
    target_size=(HEIGHT, WIDTH))

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(  
        dataframe=test,
        directory = "../input/imet-2019-fgvc6/test",    
        x_col="id",
        target_size=(HEIGHT, WIDTH),
        batch_size=1,
        shuffle=False,
        class_mode=None)


# ### Bottleneck model - VGG16 (feature extractor)

# In[6]:


model_vgg = VGG16(weights=None, include_top=False)
model_vgg.load_weights('../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


# #### Pass all the train data through the pre-trained model to extract features

# In[7]:


STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
train_data = model_vgg.predict_generator(train_generator, STEP_SIZE_TRAIN)


# #### Recreate the train labels

# In[8]:


train_labels = []
for label in train['attribute_ids'][:train_data.shape[0]].values:
    zeros = np.zeros(N_CLASSES)
    for label_i in label:
        zeros[int(label_i)] = 1
    train_labels.append(zeros)
    
train_labels = np.asarray(train_labels)


# #### Split the new train data into train and validation for the next model

# In[9]:


X_train, X_val, Y_train, Y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=0)


# ### Second model - Deep Learning MLP

# In[10]:


model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(N_CLASSES, activation="sigmoid"))

optimizer = optimizers.Adam(lr=LEARNING_RATE)
thresholds = [0.1, 0.15, 0.2, 0.25, 0.28, 0.3, 0.4, 0.5]
metrics = ["accuracy", "categorical_accuracy", f2_score_thr(0.1), f2_score_thr(0.15), f2_score_thr(0.2), 
           f2_score_thr(0.25), f2_score_thr(0.28), f2_score_thr(0.3), f2_score_thr(0.4), f2_score_thr(0.5)]
lrate = LearningRateScheduler(step_decay)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=ES_PATIENCE)
callbacks = [es]
model.compile(optimizer=optimizer, loss="binary_crossentropy",  metrics=metrics)


# In[11]:


history = model.fit(x=X_train, y=Y_train,
                    validation_data=(X_val, Y_val),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=callbacks,
                    verbose=2)


# ### Model graph loss

# In[12]:


sns.set_style("whitegrid")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex='col', figsize=(20,7))


ax1.plot(history.history['loss'], label='Train loss')
ax1.plot(history.history['val_loss'], label='Validation loss')
ax1.legend(loc='best')
ax1.set_title('Loss')

ax2.plot(history.history['acc'], label='Train Accuracy')
ax2.plot(history.history['val_acc'], label='Validation accuracy')
ax2.legend(loc='best')
ax2.set_title('Accuracy')

ax3.plot(history.history['categorical_accuracy'], label='Train Cat Accuracy')
ax3.plot(history.history['val_categorical_accuracy'], label='Validation Cat Accuracy')
ax3.legend(loc='best')
ax3.set_title('Cat Accuracy')

plt.xlabel('Epochs')
sns.despine()
plt.show()


# In[13]:


fig, axes = plt.subplots(4, 2, sharex='col', figsize=(20,7))

axes[0][0].plot(history.history['f2_score'], label='Train F2 Score')
axes[0][0].plot(history.history['val_f2_score'], label='Validation F2 Score')
axes[0][0].legend(loc='best')
axes[0][0].set_title('F2 Score threshold 0.1')

axes[0][1].plot(history.history['f2_score_1'], label='Train F2 Score')
axes[0][1].plot(history.history['val_f2_score_1'], label='Validation F2 Score')
axes[0][1].legend(loc='best')
axes[0][1].set_title('F2 Score threshold 0.15')

axes[1][0].plot(history.history['f2_score_2'], label='Train F2 Score')
axes[1][0].plot(history.history['val_f2_score_2'], label='Validation F2 Score')
axes[1][0].legend(loc='best')
axes[1][0].set_title('F2 Score threshold 0.2')

axes[1][1].plot(history.history['f2_score_3'], label='Train F2 Score')
axes[1][1].plot(history.history['val_f2_score_3'], label='Validation F2 Score')
axes[1][1].legend(loc='best')
axes[1][1].set_title('F2 Score threshold 0.25')

axes[2][0].plot(history.history['f2_score_4'], label='Train F2 Score')
axes[2][0].plot(history.history['val_f2_score_4'], label='Validation F2 Score')
axes[2][0].legend(loc='best')
axes[2][0].set_title('F2 Score threshold 0.28')

axes[2][1].plot(history.history['f2_score_5'], label='Train F2 Score')
axes[2][1].plot(history.history['val_f2_score_5'], label='Validation F2 Score')
axes[2][1].legend(loc='best')
axes[2][1].set_title('F2 Score threshold 0.3')

axes[3][0].plot(history.history['f2_score_6'], label='Train F2 Score')
axes[3][0].plot(history.history['val_f2_score_6'], label='Validation F2 Score')
axes[3][0].legend(loc='best')
axes[3][0].set_title('F2 Score threshold 0.4')

axes[3][1].plot(history.history['f2_score_7'], label='Train F2 Score')
axes[3][1].plot(history.history['val_f2_score_7'], label='Validation F2 Score')
axes[3][1].legend(loc='best')
axes[3][1].set_title('F2 Score threshold 0.5')

plt.xlabel('Epochs')
sns.despine()
plt.show()


# ### Find best threshold value

# In[14]:


best_thr = 0
best_thr_val = history.history['val_f2_score'][-1]
for i in range(1, len(metrics)-2):
    if best_thr_val < history.history['val_f2_score_%s' % i][-1]:
        best_thr_val = history.history['val_f2_score_%s' % i][-1]
        best_thr = i

threshold = thresholds[best_thr]
print('Best threshold is: %s' % threshold)


# ### Apply model to test set and output predictions

# In[15]:


test_generator.reset()
STEP_SIZE_TEST = test_generator.n//test_generator.batch_size
# Pass the test data through the pre-trained model to extract features
bottleneck_preds = model_vgg.predict_generator(test_generator, steps=STEP_SIZE_TEST)
# Make prediction using the second model
preds = model.predict(bottleneck_preds)


# In[16]:


predictions = []
for pred_ar in preds:
    valid = ''
    for idx, pred in enumerate(pred_ar):
        if pred > threshold:
            if len(valid) == 0:
                valid += str(idx)
            else:
                valid += (' %s' % idx)
    if len(valid) == 0:
        valid = str(np.argmax(pred_ar))
    predictions.append(valid)


# In[17]:


filenames = test_generator.filenames
results = pd.DataFrame({'id':filenames, 'attribute_ids':predictions})
results['id'] = results['id'].map(lambda x: str(x)[:-4])
results.to_csv('submission.csv',index=False)
results.head(10)

