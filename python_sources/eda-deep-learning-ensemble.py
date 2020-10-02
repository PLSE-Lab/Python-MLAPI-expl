#!/usr/bin/env python
# coding: utf-8

# # Deep Ensemble learning: Stacking results from deep learning models
# ### **Hyungsuk Kang, Sungkyunkwan University**
# #### 2017/10/21
# # Outline
# * **1. Introduction**
# 
# * **2. Data preparation**
# 
#     * 2.1 Load Data for Each Model
#     
#     * 2.2 Exploratory Data Analysis
#     
#     * 2.3 Feature Extraction
# 
# * **3. Training**
# 
#     * 3.1 Classifiers
#     
#     * 3.2 Evaluation
#     
#     * 3.3 Test predictions
# 
# * **4. Prediction and submission**
# 
#     * 4.1 Prediction
#     
#     * 4.2 Ensembling
#     
#     * 4.3 Results

# # **1. Introduction**
# 
# This is a full walkthrough for building the ensemble learning model for dog image dataset provided by Kaggle. Ensemble learning is a machine learning method which combines results from several model. It usually shows better results due to several reasons(reducing overfitting, etc) First, I will prepare the data (Dog images), get prediction for each model(InceptionV3, VGG, Xception), and combine each results from the model.
# 
# For more information on Keras, click this link.
# 
# # [Keras](https://keras.io/)

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from tqdm import tqdm
from keras.models import Model, Sequential
from keras.layers import Input, GlobalAveragePooling2D, Dense, Conv2D, MaxPooling2D, Dropout, Lambda, Reshape, Flatten
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
import cv2
from keras.applications.inception_v3 import preprocess_input
import matplotlib.image as mpimg
import seaborn as sns

np.random.seed(2)


# # 2. Data Preparation

# ## 2.1. Load Data for Each Model
# 
# Each model requires certain image pixel size(299x299 for inceptionV3,  Xception, and VGG)
# 
# ### Set up metadata

# In[ ]:


df = pd.read_csv('../input/dog-breed-identification/labels.csv')
df_test = pd.read_csv('../input/dog-breed-identification/sample_submission.csv')
df.head()


# In[ ]:


n = len(df)
breed = set(df['breed'])
n_class = len(breed)
class_to_num = dict(zip(breed, range(n_class)))
num_to_class = dict(zip(range(n_class), breed))


# ### Load Images

# In[ ]:


width = 299
X = np.zeros((n, width, width, 3), dtype=np.uint8) # buffer for images
y = np.zeros((n, n_class), dtype=np.uint8) # buffer for label
for i in tqdm(range(n)): # fill them up; tqdm for progress bar in loop
    X[i] = cv2.resize(cv2.imread('../input/dog-breed-identification/train/%s.jpg' % df['id'][i]), (width, width))
    y[i][class_to_num[df['breed'][i]]] = 1


# ### Load Images(test)

# In[ ]:


width = 299
n_test = len(df_test)
X_test = np.zeros((n_test, width, width, 3), dtype=np.uint8)
for i in tqdm(range(n_test)):
    X_test[i] = cv2.resize(cv2.imread('../input/dog-breed-identification/test/%s.jpg' % df_test['id'][i]), (width, width))
    


# ## 2.2 Exploratory Data Analysis
# 
# ### Distribution of Output Variable

# In[ ]:


y_eda = [list(i).index(1) for i in tqdm(y, total=n)]
g = sns.countplot(y_eda)


# It can be seen that:
#     
#     - Labels are biased to some labels(15, 68, etc).
#     - Balancing weight for each class may improve LB score.

# ## 2.3 Feature Extraction
# 
# ### Extract Bottleneck Features
# 
# Thanks to [Beluga](https://www.kaggle.com/gaborfodor), training with pretrained CNN models in Keras became possible in a Kaggle kernel.
# 
# Add the [dataset](https://www.kaggle.com/gaborfodor/keras-pretrained-models) to it and execute shell code below
# 
# Bottleneck features are saved just in case the computer is out of resources

# In[ ]:


cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)

get_ipython().system('ls ../input/keras-pretrained-models/')
get_ipython().system('cp ../input/keras-pretrained-models/* ~/.keras/models/')


# 
# Function from [Yang Peiwan's kernel](https://www.kaggle.com/yangpeiwen/keras-inception-xception-0-47):

# In[ ]:



def get_features(MODEL, data=X, batch_size=4):
    cnn_model = MODEL(include_top=False, input_shape=(width, width, 3), weights='imagenet')
    
    inputs = Input((width, width, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    cnn_model = Model(inputs, x)

    features = cnn_model.predict(data, batch_size=batch_size, verbose=1)
    return features### InceptionV3


# ### InceptionV3

# In[ ]:


inception_features = get_features(InceptionV3, X)
np.savez('bottleneck_features/inception_features.npz' , X=inception_features)


# ### Xception

# In[ ]:


xception_features = get_features(Xception, X)
np.savez('bottleneck_features/xception_features.npz' , X=xception_features)


# ### Resnet50

# In[ ]:


resnet_features = get_features(ResNet50, X)
np.savez('bottleneck_features/resnet_features.npz' , X=resnet_features)


# ### VGG19

# In[ ]:


vgg_features = get_features(VGG19, X)
np.savez('bottleneck_features/vgg_features.npz' , X=vgg_features)


# # 3. Training/Predicting Pipeline
# 
# ## 3.1. Split Train/Valid dataset

# In[ ]:


X_train_xception, X_valid_xception, y_train_xception, y_valid_xception =  train_test_split(xception_features, y, test_size=0.2, random_state=99)
X_train_inception, X_valid_inception, y_train_inception, y_valid_inception = train_test_split(inception_features, y, test_size=0.2, random_state=99)
X_train_vgg, X_valid_vgg, y_train_vgg, y_valid_vgg = train_test_split(vgg_features, y, test_size=0.2, random_state=99)
X_train_resnet, X_valid_resnet, y_train_resnet, y_valid_resnet = train_test_split(resnet_features, y, test_size=0.2, random_state=99)


# ## 3.1. Classifiers
# 
# 
# ### Neural Net

# In[ ]:


Inception_model = Sequential()
Inception_model.add(Dropout(0.2, input_shape=inception_features.shape[1:]))
Inception_model.add(Dense(n_class, activation='softmax'))

Inception_model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
           )

Inception_model.summary()


Xception_model = Sequential()
Xception_model.add(Dropout(0.2, input_shape=xception_features.shape[1:]))
Xception_model.add(Dense(n_class, activation='softmax'))

Xception_model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
           )

Xception_model.summary()


VGG_model = Sequential()
VGG_model.add(Dropout(0.2, input_shape=vgg_features.shape[1:]))
VGG_model.add(Dense(n_class, activation='softmax'))

VGG_model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'], 
           )

VGG_model.summary()

Resnet_model = Sequential()
Resnet_model.add(Dropout(0.2, input_shape=resnet_features.shape[1:]))
Resnet_model.add(Dense(n_class, activation='softmax'))

Resnet_model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
           )

Resnet_model.summary()


# ### Callbacks
# 
# - ModelCheckpoint is used to get the best model after epochs
# 
# - ReduceLROnPlateau is used to manipulate learning rate for more delciate correction

# In[ ]:


inception_callbacks=[ReduceLROnPlateau(monitor='acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001),
                      ModelCheckpoint(filepath='saved_models/inception.best.from_features.hdf5', 
                               verbose=1, save_best_only=True)
                     ]

xception_callbacks=[ReduceLROnPlateau(monitor='acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001),
                      ModelCheckpoint(filepath='saved_models/xception.best.from_features.hdf5', 
                               verbose=1, save_best_only=True)
                     ]

resnet_callbacks=[ReduceLROnPlateau(monitor='acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001),
                      ModelCheckpoint(filepath='saved_models/resnet.best.from_features.hdf5', 
                               verbose=1, save_best_only=True)
                     ]

vgg_callbacks=[ReduceLROnPlateau(monitor='acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001),
                      ModelCheckpoint(filepath='saved_models/vgg.best.from_features.hdf5', 
                               verbose=1, save_best_only=True)
                     ]


# In[ ]:


epochs = 1 # Increase this if you want more accurate results(It is recommended to run on personal computer in this case)

from sklearn.utils import class_weight

class_weight = class_weight.compute_class_weight('balanced', np.unique(y_eda), y_eda)

inception_history = Inception_model.fit(X_train_inception, y_train_inception, 
          validation_data=(X_valid_inception, y_valid_inception),
          epochs=epochs, 
          callbacks=inception_callbacks,
          class_weight=class_weight,
          batch_size=8, verbose=1)

xception_history = Xception_model.fit(X_train_xception, y_train_xception, 
          validation_data=(X_valid_xception, y_valid_xception),
          epochs=epochs,                            
          callbacks=xception_callbacks,
          class_weight=class_weight,
          batch_size=8, verbose=1)

resnet_history = Resnet_model.fit(X_train_resnet, y_train_resnet, 
          validation_data=(X_valid_resnet, y_valid_resnet),
          epochs=epochs, 
          callbacks=resnet_callbacks,
          class_weight=class_weight,
          batch_size=8, verbose=1)

vgg_history = VGG_model.fit(X_train_vgg, y_train_vgg, 
          validation_data=(X_valid_vgg, y_valid_vgg),
          epochs=epochs, 
          callbacks=vgg_callbacks,
          class_weight=class_weight,
          batch_size=8, verbose=1)


#  ## 3.2 Evaluation

# ### Learning Curve
# 
# Learning rate is the step by which the optimizer walks through the 'loss landscape'. The higher it is, the bigger are the steps and the quicker is the convergence. However, the sampling is very poor with an high LR and the optimizer could probably fall into a local minima. Low learning rate shows slower convergence and lower chance of falling into a local minima, but it leads to underfitting and requires more epochs.
# To detect this, learning curve plot is used. 
# 
# 
# Picture below shows each case of the learning rate's status based on loss:
# 
# ![Learning Curve](http://img1.imagilive.com/0717/learningrates.jpg)
# 
# 
# For accuracy:
# ![Learning Curve](http://img1.imagilive.com/0717/accuracies1de.jpg)

# In[ ]:


# Plot the loss and accuracy curves for training and validation on InceptionV3
fig, ax = plt.subplots(2,1)
ax[0].plot(inception_history.history['loss'], color='b', label="Training loss")
ax[0].plot(inception_history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(inception_history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(inception_history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# In[ ]:


# Plot the loss and accuracy curves for training and validation on xception model
fig, ax = plt.subplots(2,1)
ax[0].plot(xception_history.history['loss'], color='b', label="Training loss")
ax[0].plot(xception_history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(xception_history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(xception_history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# In[ ]:


# Plot the loss and accuracy curves for training and validation on resnet model
fig, ax = plt.subplots(2,1)
ax[0].plot(resnet_history.history['loss'], color='b', label="Training loss")
ax[0].plot(resnet_history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(resnet_history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(resnet_history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# In[ ]:


# Plot the loss and accuracy curves for training and validation on vgg model
fig, ax = plt.subplots(2,1)
ax[0].plot(vgg_history.history['loss'], color='b', label="Training loss")
ax[0].plot(vgg_history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(vgg_history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(vgg_history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# ### Confusion Matrix 
# 
# Confusion matrix can check false positives for each labels.
# 
# This can visualize bias and variance of the model's prediction.

# ### InceptionV3

# In[ ]:


# Look at confusion matrix 
from sklearn.metrics import confusion_matrix

# Predict the values from the validation dataset
Y_pred = Inception_model.predict(X_valid_inception)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_valid_inception,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
ax = sns.heatmap(confusion_mtx)


# ### Xception

# In[ ]:


# Look at confusion matrix 
from sklearn.metrics import confusion_matrix

# Predict the values from the validation dataset
Y_pred = Xception_model.predict(X_valid_xception)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_valid_xception,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
ax = sns.heatmap(confusion_mtx)


# ### Resnet50

# In[ ]:


# Look at confusion matrix 
from sklearn.metrics import confusion_matrix

# Predict the values from the validation dataset
Y_pred = Resnet_model.predict(X_valid_resnet)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_valid_resnet,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
ax = sns.heatmap(confusion_mtx)


# ### VGG19

# In[ ]:


# Look at confusion matrix 
from sklearn.metrics import confusion_matrix

# Predict the values from the validation dataset
Y_pred = VGG_model.predict(X_valid_vgg)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_valid_vgg,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
ax = sns.heatmap(confusion_mtx)


# # 4. Prediction and Submission

# ## 4.1 Prediction

# In[ ]:


inception_features_test = get_features(InceptionV3, X)
np.savez('bottleneck_features/inception_features_test.npz' , X=inception_features_test)


# In[ ]:


xception_features_test = get_features(Xception, X)
np.savez('bottleneck_features/xception_features_test.npz' , X=xception_features_test)


# In[ ]:


resnet_features_test = get_features(ResNet50, X)
np.savez('bottleneck_features/resnet_features_test.npz' , X=resnet_features_test)


# In[ ]:


vgg_features_test = get_features(VGG19, X)
np.savez('bottleneck_features/vgg_features_test.npz' , X=vgg_features_test)


# ### Load weights with the highest accuracy results

# In[ ]:


Inception_model.load_weights('saved_models/inception.best.from_features.hdf5')
Xception_model.load_weights('saved_models/xception.best.from_features.hdf5')
Resnet_model.load_weights('saved_models/resnet.best.from_features.hdf5')
VGG_model.load_weights('saved_models/vgg.best.from_features.hdf5')


# In[ ]:


y_pred = Inception_model.predict(inception_features_test, batch_size=128)
for b in breed:
    df_test[b] = y_pred[:,class_to_num[b]]
inception_test = df_test.copy()
df_test.to_csv('pred_inception.csv', index=None)

y_pred = Xception_model.predict(xception_features_test, batch_size=128)
for b in breed:
    df_test[b] = y_pred[:,class_to_num[b]]
xception_test = df_test.copy()
df_test.to_csv('pred_xception.csv', index=None)

y_pred = Resnet_model.predict(resnet_features_test, batch_size=128)
for b in breed:
    df_test[b] = y_pred[:,class_to_num[b]]
resnet_test = df_test.copy()
df_test.to_csv('pred_resnet.csv', index=None)

y_pred = VGG_model.predict(vgg_features_test, batch_size=128)
for b in breed:
    df_test[b] = y_pred[:,class_to_num[b]]
vgg_test = df_test.copy()
df_test.to_csv('pred_vgg.csv', index=None)


# ## 4.2 Ensembling
# 
# I used stacking method for ensembling the results from the models.

# In[ ]:


n_model = 4
id_test = inception_test['id']
sum_test = inception_test.drop('id', axis=1) + xception_test.drop('id', axis=1) + resnet_test.drop('id', axis=1) + vgg_test.drop('id', axis=1)
ensemble_test = (np.exp(sum_test / n_model) - 1)
ensemble_test.insert(0, 'id', id_test)

ensemble_test.to_csv('pred_stacked.csv', index=None)


# ## 4.3 Results

# 1. Inception LB 0.73604
# 2. Xception LB 0.57665
# 3. Resnet LB 4.62757 (This one had the fuzziest confusion matrix, needs a lot of epochs or bigger learning rate to prevent underfitting)
# 4. VGG19 LB 1.54441 (This one too needs more adjustments than others)
# 5. Stacked LB 0.85407
# 
# Fine tune these models then you will get better result
