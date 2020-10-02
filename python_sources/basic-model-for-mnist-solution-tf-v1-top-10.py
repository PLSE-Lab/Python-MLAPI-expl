#!/usr/bin/env python
# coding: utf-8

# # ** Basic Model for MNIST Solution, TF-v1, Top 10 **
# 
# Author: CY Peng
# 
# - First Release: 2019/9/23, Transfer Leaerning Model: Required More Training
# - Second Release: 2019/9/24, Bug Fixed 
# - Third Release: 2019/9/25, VGG-Like Model Solution
# - Fourth Test: 2020/1/2, Using Tenflow-v1 to Established Model

# This project, we cover several part:
# 
# - Preparation
#   - Tool Liberary Usage
#   - Data Preparation
# - Define the Problem
# - Explore Data Analysis, EDA
#   - Preliminary Investigation
#   - Statistical Information
# - Feature Engineering
#   - Gray Images Conversion
#   - Data Augmentation
# - Model Established
#   - LeNet-5 Model
#   - VGG-Like Model
#   - Model Selection
# - Model Prediction

# # 1. Preparation
# 
# ## Tool Liberary Usage
# First, we import the libraries for this project. According to the data science, there are five steps to analyze for this project:
# - Step 1: Define the Problem
# - Step 2: Explore Data Analysis, EDA: we use the numpy, pandas and matplotlib liberaries to analyze the data.
# - Stpe 3: Feature Engineering: we use the numpy, pandas and the cv2 to process the data
# - Step 4: Model Established: we use the tensorflow to established the model
# - Step 5: Model Maintenance: this step, we use the model to predict the unknown data
# 
# Here, we import these libraries:

# In[ ]:


# Data Processing
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns  # data visualization
import plotly.express as px # data visualization
import plotly.graph_objs as go # go object plot

# Image Processing
#import random # Data Augment
#import cv2 # Image Preprocessing

# Machine Learning
from sklearn.preprocessing import OneHotEncoder # Output Preprocessing
from sklearn.model_selection import train_test_split # Machine Learning Data Preprocessing
#!pip install tensorflow --upgrade
#!pip install -U tensorflow-gpu==2.0.0
#!pip install -q tf-nightly-2.0-preview
#!pip install kerastuner
import tensorflow as tf # Machine Learning DL Module
from tensorflow import keras # Machine Learning DL Module
from tensorflow.keras.models import Sequential # Machine Learning DL Module
from tensorflow.keras.layers import Dense, MaxPool2D, Flatten, Conv2D, Dropout # Machine Learning DL Module
from tensorflow.compat.v2.keras.utils import to_categorical # Output Processing
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Data Augmentation
from tensorflow.keras import regularizers
#from tensorflow.keras.utils import multi_gpu_model
#from tensorboard.plugins.hparams import api as hp
#from kerastuner.tuners import GridSearch #Hyperparameters
#from kerastuner.distributions import Range, Choice #Hyperparameters
#from tensorboard.plugins.hparams import api as hp #Hyperparameters
from sklearn.model_selection import GridSearchCV #Hyperparameters
#!pip install talos
#import talos #Hyperparameters
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

#from tensorflow.contrib.opt import AdamWOptimizer 
#from tensorflow.python.keras.optimizers import TFOptimizer

print("Tensorflow DL Version: " + tf.__version__)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
"""
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""
# Any results you write to the current directory are saved as output.
print("Setup Completed")


# ## Data Preparation
# Here, we read the train and prediction file.

# In[ ]:


train_file = "../input/digit-recognizer/train.csv"
predict_file = "../input/digit-recognizer/test.csv"
submission_file = "../input/digit-recognizer/sample_submission.csv"

#X, y = prep_data(train_file)
train_data = pd.read_csv (train_file, sep=',') 
predict_data = pd.read_csv (predict_file, sep=',') 
print("Files Preparation Completed")


# # 2. Define Problem
# Based on the problem description, we summarize as following as descriptions:
# 
# - Problem Target: Multi-Classifications, digital 0-9
# - Train File: 251.mb data, each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. There are 785 columns, including the 784 pixels and 1 label.
# - Prediction File: 167.5 mb data, 784 pixels for each image
# 
# # 3. Explore Data Analysis, EDA
# ## Preliminary Investigation
# First, we need to know some simple information about the files:

# In[ ]:


print('**************************Train File Preliminary Investigation**************************')
print('1. Train File Shape:', train_data.shape)
missing_val_count_by_column = (train_data.isnull().sum())
print('2. Train File Missing Valu:', missing_val_count_by_column[missing_val_count_by_column > 0])
print("   Missing Values in Train File: {}".format(train_data.isna().any().any()))
print('3. Train File Variables Information:')
print(train_data.info())
print('4. Train File Variables Unique Number:')
print(train_data.nunique())
print('                                                                                       ')
print('**************************Prediction File Preliminary Investigation**************************')
print('1. Test File Shape:', predict_data.shape)
missing_val_count_by_column = (predict_data.isnull().sum())
print('2. Test File Missing Value:', missing_val_count_by_column[missing_val_count_by_column > 0])
print("   Missing Values in Test File: {}".format(predict_data.isna().any().any()))
print('3. Test File Variables Information:')
print(predict_data.info())
print('4. Test File Variables Unique Number:')
print(predict_data.nunique())


# Of the above investigation, there are some tips for two files:
# 
# - From 1., there are 42000 images in the train file and 28000 images in the prediction file
# - From 2., we know that there are no missing values in the files! So, don't descard any image.
# - From 3., all variables type are integer, we need to convert the array to the gray image matrix.
# - From 4., we know that the file image pixel ID is from 0 to 783. Label is the digital 0-9, the problem is the multi-classification.
# 
# We set some parameters and check the data type:

# In[ ]:


def datatype_check(data, type_list):
    for type_name in type_list:
        try:
            temp = data[type_name]
        except:
            print('Some Issues in Data Type: {}!'.format(type_name))
            pass
    print('Data Type Check OK!')

# Parameters Setting
img_rows, img_cols = 28, 28
num_classes = 10
output_list = ['label']
print("Parameters Setting OK")

print('**************************Output Data List**************************')
print(output_list)
datatype_check(train_data, output_list)


# Finally, we check the some data for the train and the test files:

# In[ ]:


print('Train File:')
train_data.head().T


# In[ ]:


print('Prediction File:')
predict_data.head().T


# ## Statistical Information
# Here, we show the distribution plot for the data. First, we write the some plot as following as code:

# In[ ]:


# Count Plot
def count_plot(data, type_list, target_list):
    for type_name in type_list:
        for target_name in target_list:
            plt.figure()
            if type_name == target_name:
                sns.countplot(x= type_name, data=data)
            else:
                sns.countplot(x= type_name, hue=target_name, data=data)
        
print("Function Read Completed")


# Count Plot:

# In[ ]:


print("Train File:")
count_plot(train_data, ['label'], ['label'])


# From above plot, we know that the each data type are around 4000 data. According to the statistics theory, there are at least 5000 data to established the good model (statistic model); therefore, we want to use the data augmentation technology to increase the different images data as following feature engineering. 

# # 4. Feature Engineering
# According to the EDA, we have to use the two methods to established the recognition model. First, we will convert the image array to the gray image matrix. Second, we use the data augmentation technolgy to increase the images data.
# 
# ## Gray Images Conversion
# First, we need to convert the array to the gray image matrix as following as Library:

# In[ ]:


def label_conversion(data, num_classes = 10):
    label_array = np.array(data, dtype='uint8') #pd.get_dummies(data['label'])
    label_array = to_categorical(label_array,num_classes=num_classes)
    return label_array

def images_conversion(data, img_row = 28, imag_col = 28):
    image_array = np.array(data, dtype='uint8')
    image_array = np.reshape(image_array, (len(image_array), img_row, imag_col, 1))
    image_array = image_array * (1. / 255) - 0.5
    image_array = np.reshape(image_array, (len(image_array), img_row, imag_col, 1))
    
    return image_array

print("Function Established Completed")


# Data conversion from the train file and the prediction file:

# In[ ]:


y = label_conversion(train_data[output_list[0]], num_classes = num_classes)
X = images_conversion(train_data.drop(output_list[0], axis=1), img_row = img_rows, imag_col = img_cols)
X_predict = images_conversion(predict_data, img_row = img_rows, imag_col = img_cols)
print("Conversion Completed")


# We want to check the data:

# In[ ]:


print("Training Data: {} \n  Labels: {}".format(X, y))


# In[ ]:


print("Prediction Data: {} \n".format(X_predict))


# ## Data Augmentation
# Second, we use the tensorflow keras library to do the images augmentation as following as setting:

# In[ ]:


# Data Augmentation Setting 
data_aug = ImageDataGenerator(
           featurewise_center=False,  # set input mean to 0 over the dataset
           samplewise_center=False,  # set each sample mean to 0
           featurewise_std_normalization=False,  # divide inputs by std of the dataset
           samplewise_std_normalization=False,  # divide each input by its std
           zca_whitening=False,  # apply ZCA whitening
           rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
           zoom_range = 0.1, # Randomly zoom image 
           width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
           height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
           horizontal_flip=False,  # randomly flip images
           vertical_flip=False)  # randomly flip images
#data_aug.fit(X_train)

print("Data Augmentation Setting Completed")


# # 5. Model Establihsed
# In this section and next section, we show the how to use the LeNet-5 Model and the VGG-Like Model to train, test and predict the unknown data.
# First, we need to prepare the data for training; then, we want to established the model as following.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
print("Data Preparation Completed")


# We set the some parameters:

# In[ ]:


# Parameters Setting
l_model_name = 'lenet5_model' # Original Model
v_model_name = 'vgg_model' # Transfer Learning Model
batch_size = 64
num_classes = 10
epoch_num = 6
dropout_rate = 0.5

# adam, sgd with weight decay
oz = keras.optimizers.Adam(lr=0.001, 
                           beta_1=0.9, 
                           beta_2=0.999, 
                           epsilon=1e-08, 
                           decay=1e-4, 
                           amsgrad=False)
sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


# Hyper Parameters Setting
#HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
#HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.5, 0.8))
#HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

print("Parameters Setting Completed")


# ## LeNet-5 Model 
# This Project, we use the LeNet-5 model to solve the problem. LeNet-5 paper is the fundemental CNN in Deep Learning, along side that paper. In this project, we will use the original LeNet-5 model to solve the problem, and then using VGG-like model by learning to solve the problem. We will select the best model from these models to predict the unknown data.
# 
# According to the reference (Ref to [my GitLab](https://gitlab.com/pcyslm/tf_mnistrecognition_rd), using the TensorFlow module to solve the MNIST problem), there are three type layer in the LeNet-5:
# - Convolution Layer: The main function is features extraction; there are two features, local receptive fields and shared weights.
# - Max Pooling Layer: The main function is subsampling; down the sensitivity of the shift/scale/distortion invariance of the output.
# - Fully Connected Layer, FC: Also called multi-layer perceptrons, the main function is classfication or regression, but it has high degree sensitivity of the input
# 
# We set the callback monitor to save the model; then, define the LeNet-5 model:

# In[ ]:


# Callback Setting
# Save the Best Model as High val_acc
l_ck_callback = tf.keras.callbacks.ModelCheckpoint(l_model_name+'_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc', mode='max',
                                                  verbose=2, save_best_only=True, save_weights_only=True)
l_best_callback = tf.keras.callbacks.ModelCheckpoint(l_model_name+'_weights.h5', monitor='val_acc', mode='max',
                                                  verbose=2, save_best_only=True, save_weights_only=True)
# Tensorboard Monitor
l_tb_callback = tf.keras.callbacks.TensorBoard(log_dir='lenet_logs')
# Adjusting the Learning Rate for the each epoch if the val_loss don't drop down
# learning rate = factor * lr_old
l_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=2, factor=0.5, min_lr=0.0001)

# LeNet-5 Model Established
lenet5_model = Sequential(name=l_model_name)
lenet5_model.add(Conv2D(16, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
lenet5_model.add(MaxPool2D(pool_size=(2, 2),
                 strides=None,
                 padding='valid'))
lenet5_model.add(Conv2D(16, kernel_size=(5, 5),
                 activation='relu'))
lenet5_model.add(MaxPool2D(pool_size=(2, 2),
                 strides=None,
                 padding='valid'))
lenet5_model.add(Flatten())
lenet5_model.add(Dense(120, input_shape=(400,), activation='relu'))
lenet5_model.add(Dense(84, input_shape=(120,), activation='relu'))
lenet5_model.add(Dense(num_classes, input_shape=(84,), activation='softmax'))

lenet5_model.compile(loss=keras.losses.categorical_crossentropy,
                     optimizer=oz,
                     metrics=['accuracy', 'categorical_crossentropy'])
lenet5_model.summary()


# Training stage:

# In[ ]:


# Training/ Testing
lenet5_model_history = lenet5_model.fit_generator(data_aug.flow(X_train, y_train, batch_size=batch_size),
                                                  epochs = epoch_num, 
                                                  validation_data = (X_test, y_test),
                                                  verbose = 2, 
                                                  steps_per_epoch=X_train.shape[0],
                                                  callbacks=[l_tb_callback, 
                                                             l_lr_callback, 
                                                             l_ck_callback, 
                                                             l_best_callback])
lenet5_model.save(l_model_name+'.h5')
del lenet5_model
print("Training Completed!")


# ## VGG-Like Model
#  According to the [keras official web](https://keras.io/getting-started/sequential-model-guide/), we use the VGG-Like model to solve the problem. VGG Model is the first using the droup out layer, and using more hidden layers to training data relative LeNet-5 model. Similarly, we set the parameters for callback, and then define the model.

# In[ ]:


# Callback Setting
# Save the Best Model as High val_acc
v_ck_callback = tf.keras.callbacks.ModelCheckpoint(v_model_name+'_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc', mode='max',
                                                  verbose=2, save_best_only=True, save_weights_only=True)
v_best_callback = tf.keras.callbacks.ModelCheckpoint(v_model_name+'_weights.h5', monitor='val_acc', mode='max',
                                                  verbose=2, save_best_only=True, save_weights_only=True)
# Tensorboard Monitor
v_tb_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
# Adjusting the Learning Rate for the each epoch if the val_loss don't drop down
# learning rate = factor * lr_old
v_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=2, factor=0.5, min_lr=0.0001)

# Hyperparameters Monitor
"""
METRIC_ACCURACY = 'accuracy'
with tf.summary.create_file_writer('logs/vgg_hparam_tuning').as_default():
    hp.hparams_config(
    hparams=[HP_DROPOUT, HP_OPTIMIZER], #HP_NUM_UNITS, 
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )
"""

# VGG-Like Model:
vgg_model = Sequential(name=v_model_name)

vgg_model.add(Conv2D(32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape=(img_rows, img_cols, 1)))
vgg_model.add(Conv2D(32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
vgg_model.add(MaxPool2D(pool_size=(2,2)))
vgg_model.add(Dropout(dropout_rate*0.5))

vgg_model.add(Conv2D(64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
vgg_model.add(Conv2D(64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
vgg_model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
vgg_model.add(Dropout(dropout_rate*0.5))

vgg_model.add(Flatten())
vgg_model.add(Dense(256, activation = "relu"))
vgg_model.add(Dropout(dropout_rate))
vgg_model.add(Dense(num_classes, activation = "softmax"))

vgg_model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=sgd,
                  metrics=['accuracy', 'categorical_crossentropy'])
vgg_model.summary() 


# Next, we will excute the training stage:

# In[ ]:


# Training/ Testing
vgg_model_history = vgg_model.fit_generator(data_aug.flow(X_train, y_train, batch_size=batch_size),
                                            epochs = epoch_num, 
                                            validation_data = (X_test, y_test),
                                            verbose = 2, 
                                            steps_per_epoch=X_train.shape[0],
                                            callbacks=[v_tb_callback, 
                                                       v_lr_callback, 
                                                       v_ck_callback, 
                                                       v_best_callback])
vgg_model.save(v_model_name+'.h5')
del vgg_model
print("Training Completed!")


# ## Model Selection
# We will select the best model from the original LeNet-5 Model and the modified model from the learning curve as following:

# In[ ]:


def plot_history(histories, key='categorical_crossentropy', epoch = 3):
    epoch_array = range(epoch)
    plt.figure(figsize=(16,10))
    for name, history in histories:
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(epoch_array, history['acc'], label='Training Accuracy')
        plt.plot(epoch_array, history['val_acc'], label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        #plt.ylim([min(plt.ylim()),1])
        plt.grid()
        plt.title(name.title()+' Training and Validation Accuracy')
        
        plt.subplot(2, 1, 2)
        plt.plot(epoch_array, history['loss'], label='Training Loss')
        plt.plot(epoch_array, history['val_loss'], label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        #plt.ylim([0,1.0])
        plt.grid()
        plt.title(name.title()+' Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()
        """
        val = plt.plot(epoch_array, history['val_'+key],
                       '--', label=name.title()+' Val')
        plt.plot(epoch_array, history[key], color=val[0].get_color(),
                 label=name.title()+' Train')
        plt.xlabel('Epochs')
        plt.ylabel(key.replace('_',' ').title())
        plt.legend()
        plt.xlim([0,max(epoch_array)])
        plt.grid()
        """   

#print(lenet5_model_history.history.keys())
plot_history([(l_model_name, lenet5_model_history.history),
              (v_model_name, vgg_model_history.history)], epoch = epoch_num)


# The best model is:

# In[ ]:


def best_model(histories, key='categorical_crossentropy'):
    ini_flag = bool(1)
    best_model = None
    best_score = 0
    for name, history in histories:
        val_score_record = history['val_'+key]
        for score in val_score_record:
            if ini_flag:
                best_score = score
                best_model = name
                ini_flag = bool(0)
            
            if best_score > score:
                best_score = score
                best_model = name
        
    return best_model

best_model_name = best_model([(l_model_name, lenet5_model_history.history),
                             (v_model_name, vgg_model_history.history)])
print(best_model_name)


# According to the learning curve, the performance of the VGG-Like model is better than the LeNet-5 model. 

# In[ ]:


# Hyperparameters Monitor
#HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
"""
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.5, 0.8))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
METRIC_ACCURACY = 'accuracy'

with tf.contrib.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_DROPOUT, HP_OPTIMIZER], #HP_NUM_UNITS, 
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )
"""
"""
param_grid = dict(optimizer = ['adam', 'sgd'], dropout_rate = [0.5, 0.8])
hyper_best_model_name = 'best_model'

# Model Load
load_model = keras.models.load_model(best_model_name +'.h5')
load_model.load_weights(best_model_name + '_weights.h5')
print("Check Model:")
load_model.evaluate(X_test, y_test)
#load_model.compile(loss=keras.losses.categorical_crossentropy,
#                   optimizer=param_grid, 
#                   metrics=['accuracy', 'categorical_crossentropy'])
load_model.summary()

# Grid Search
keras_model = KerasClassifier(build_fn=load_model, verbose = 3) #build_fn=
grid_search_model = GridSearchCV(estimator=keras_model, param_grid=param_grid, n_jobs=-1, cv = 5)
#print(type(grid_search_model))
grid_search_model.fit(X_train, y_train)
#load_model.save(hyper_best_model_name+'.h5')
grid_result.best_estimator_.model.save(grid_best_model_name)
"""


# # 6. Model Prediction
# In this section, we use the best model to predict the unknown images data:

# In[ ]:


# Load Model
load_model = keras.models.load_model(best_model_name+'.h5')
#load_model.load_weights(best_model_name + '_weights.h5')
print("Check Model:")
load_model.evaluate(X_test, y_test)

# Submission File Read
sample_submission = pd.read_csv(submission_file)
submission_id = sample_submission["ImageId"]

# Submission File Prediction
#print(np.argmax(load_model.predict(X), axis=1))
submission = pd.DataFrame({
    "ImageId": submission_id,
    "Label": np.argmax(load_model.predict(X_predict), axis=1)
    })
submission.to_csv('submission.csv', index=False)
print("Submission File Produced Completed")


# Check the submission file:

# In[ ]:


submission.head()


# If you have any questions or suggestions, please let me know, thank you. Enjoy it!
