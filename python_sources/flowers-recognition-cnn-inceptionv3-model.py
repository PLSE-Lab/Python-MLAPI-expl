#!/usr/bin/env python
# coding: utf-8

# # Intro
# Refer to one of classical classification problems - flowers recognition and try to address it using on of Machine Learning methods.
# 
# We choose a CNN network since this neural networks are indistrial standard in computer vision. Among many CNN architectures we choose Inception v3 [1] for learning purpose.  
# 
# Let's get down to our task and remember: the days are long, the years - short.

# # Import libraries and tools

# In[ ]:


# Import libraries and tools
# Data preprocessing and linear algebra
import os, re, random
from os.path import join
import zipfile
from pathlib import Path
import shutil
from sklearn.datasets import load_files
import pandas as pd
import numpy as np
np.random.seed(2)

# Visualisation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Tools for cross-validation, error calculation
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical

# Machine Learning
from keras.models import Model
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, Activation
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import optimizers
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# # Data load

# The way in which we plan to pass our images to the input of the model depends on the data organization within directories. Or vice versa - a dichotomous question.
# There are two main options for organizing images in the original dataset.
# 
# **1.Images are sorted into folders according to classes.**   
# For example, images of cars - in the folder "Cars", images of trains - in the folder "Trains". This is the most convenient structure for passing to the model input.
# 
# In this case, the standard Keras ImageDataGenerator() method is suitable. However, in order to make it able to "see" the images, we will need to create a subdirectory inside each directory and moving the images into it. In Kaggle environment it means that we need to create appropriate subfolder hierarchy in ../output and then copy all images from ../input into it, not forgetting to divide them into the train, valid and test subfolders.
# 
# **2.Images are interspersed and a metadata file with description attached.**
# 
# In this case there are few ways to go for.  
# The first: to write a small python script, which yields batches of images and labels while reading the csv.  
# The second: to write your custom method, which extracts images from their subfolders, assign labels to them, and finally writes them down into common X and Y subsets. (Truth be sayed, this approach is universal and also can be used in case of dataset, devided on category subfolders).    
# The third - use another high-level API such as Dataset.

# ### Load images into array

# In[ ]:


print(os.listdir('../input/'))


# In[ ]:


INPUT_PATH = '../input/flowers-recognition/flowers/flowers/'
print(os.listdir(INPUT_PATH))


# In[ ]:


img_folders = [join(INPUT_PATH, dir) for dir in os.listdir(INPUT_PATH)]
list(img_folders)


# In[ ]:


# Load images into NumPy array
images = load_files(INPUT_PATH, random_state=42, shuffle=True)
X = np.array(images['filenames'])
y = np.array(images['target'])
labels = np.array(images['target_names'])

# Remove unnecessary .pyc and .py files
pyc_file = (np.where(file==X) for file in X if file.endswith(('.pyc','.py')))
for i in pyc_file:
    X = np.delete(X, i)
    y = np.delete(y, i)


# In[ ]:


# Our array summary
print(f'Target labels (digits) - {y}')
print(f'Target labels (names) - {labels}')
print(f'Number of uploaded images : {X.shape[0]}')


# In[ ]:


# Draw random image directly from dataset for aesthetic reasons only
img = plt.imread('../input/flowers-recognition/flowers/daisy/100080576_f52e8ee070_n.jpg')
plt.imshow(img);


# In[ ]:


# Check our target y variable
flowers = pd.DataFrame({'species': y})
flowers.count()


# In[ ]:


# Correspond species and flowers and form digit labels
flowers['flower'] = flowers['species'].astype('category')
labels = flowers['flower'].cat.categories


# In[ ]:


labels


# In[ ]:


# Let's implement a constant - standard image size for Inception model input, which is 150 px
image_size = 150


# In[ ]:


# Write images into NumPy array using sklearn's img_to_array() method
def imageLoadConverter(img_paths):
    # Load
    images = [load_img(img_path, target_size=(image_size, image_size)) for img_path in img_paths]
    # Write into array
    images_array = np.array([img_to_array(img) for img in images])
    
    return(images_array)

# Convert into NumPy array
X = np.array(imageLoadConverter(X))
# Print result
print(f'Function worked with following output (images, width, height, color): {X.shape}')


# ### Label encoding

# We have 5 classes of species and 5 labels for each of them (0,1,2,3,4,5). In order to pass them on network inputs we should make some preparation known as One-Hot Encoding, which takes a single integer and produces a vector where a single element is 1 and all other elements are 0, like [0, 1, 0, 0].  
# There are several ways in Python to do it, we will choose Keras's to_categorical() popular implementation.

# In[ ]:


# Convert classes in digit form
num_classes = len(np.unique(y))
print(f'Classes: {num_classes} and corresponding labels: {labels}')


# In[ ]:


# One-Hot Encoding
y = to_categorical(y, num_classes)
print(y.shape)


# ### Split data on train and validation subsets ####

# In[ ]:


# Split data on train, validation and test subsets
# Using 10% or 20% from train data is classical approach

# First, split X into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=2)

# Second, split test into test and validation subsets in equal proportion
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True, random_state=2)


# In[ ]:


# Count number of elements in subsets
total_X_train = X_train.shape[0]
total_X_val = X_val.shape[0]
total_X_test = X_test.shape[0]


# In[ ]:


print(f'Train: {total_X_train}')
print(f'Validation: {total_X_val}')
print(f'Test: {total_X_test}')


# In[ ]:


# Delete X since it will not be needed further
del X


# # Machine learning

# ### Model choose

# Lets:   
# a) choose **Inception V3** model which is one of the ImageNet winners with very high accuracy and low computer resources;  
# b) use a Transfer Learning paradigm.  
# 
# Transfer learning refers to a technique for predictive modeling on a different but somehow similar problem that can then be reused to accelerate the training and improve the performance of a new model. If we examine the process of **deep learning**, it becomes clear that on the first layers network is trying to grasp the most basic laws (edges), on the middle layers - the most important (shapes), and on the last layer - specific details (high level features).
# 
# In deep learning, this means reusing the weights in one or more layers from a pre-trained network model in a new model and either keeping the weights fixed, fine tuning them, or adapting the weights entirely when training the model [2]
# 
# Pre-trained models trained with a million level data by hundreds of researchers having strong computation power (CPU, GPU and TPU). We can use pre-trained model and train it with a small set of our data. The trick is here to freeze or lock the early layers and let the final layers to be trained. In this way, our new model can know the facial patterns in middle and high level features as well.

# ### Model description

# Inception V3 [1] is a type of Convolutional Neural Networks.  It consists of many convolution and max pooling layers and includes fully connected neural networks. Network's conceptual scheme:
# ![image.png](attachment:image.png)

# ### Model implementation

# In[ ]:


# By default, the InceptionV3 model expects images as input with the size 150x150 px with 3 channels
input_shape = (image_size, image_size, 3)


# In[ ]:


# Define model constants
batch_size = 8
epochs = 20


# In[ ]:


# Define our pre-trained model, downloading weights from Imagenet
# pre_trained_model = InceptionV3(input_shape = input_shape, include_top = False, weights = 'imagenet')

# Define our pre-trained model, using weights, uploaded from Kaggle's Keras Inception dataset
local_weights = "../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
pre_trained_model = InceptionV3(input_shape = input_shape, include_top = False, weights = None)


# In[ ]:


# Load weights into network
pre_trained_model.load_weights(local_weights)


# A few words about models parameters.  
# **input_shape:** self-descriptive, defined earlier as images with, height and color code;  
# **include_top = False:** we are going to use all the layers in the model except for the last fully connected layer as it is specific to the ImageNet competition;  
# **weights = 'imagenet':** download pre-trained weights trained on Imagenet extra-big dataset.  
# **weights = None**: load pure model and then upload weights from local machine.
# 

# In[ ]:


# Print models summary table
print(pre_trained_model.summary())


# In[ ]:


# Print number of models layers
len(pre_trained_model.layers)


# In[ ]:


# Set layers to be not trainable since they are already are
for layer in pre_trained_model.layers:
     layer.trainable = False


# Our pre-trained model goes without top and output layers, we should specify them manually.
# 
# Usually Flatten() or GlobalAveragePoolingXD() layers are placed at the end of the CNN to get a shape that works with dense layers. What is the difference between them?  
# 
# **Flattening** a tensor means to remove all of the dimensions except for one. In other words, a flatten operation on a tensor *reshapes* the tensor to have the shape that is equal to the number of elements contained in tensor non including the batch dimension.  
# 
# **GlobalAveragePooling** is a methodology used for better representation of your vector. It can be 1D/2D/3D. The main idea is to pool the data by averaging it (GlobalAveragePooling) or picking maximum value (GlobalMaxPooling). Padding is required to take the corner cases into the account.

# In[ ]:


# Add custom layers
x = pre_trained_model.output
# Add Pooling layer
x = Flatten()(x)
# Add a fully connected layer with 1024 nodes and ReLU activation
x = Dense(1024, activation="relu")(x)
# Add a dropout with rate 0.5
x = Dropout(0.2)(x)
# Specify final output layer with SoftMax activation
predictions = Dense(5, activation="softmax")(x)


# A few notes about dropout rate. We empyrically choosed 0,2 rate since typical 0,5 given less accuracy. A good and gentle dropout rate tuning can be found in [3].

# In[ ]:


pre_trained_model.input


# In[ ]:


predictions


# In[ ]:


# Build the final model 
inception_model = Model(inputs=pre_trained_model.input, 
                        outputs=predictions
                       )


# In[ ]:


# Compile model
inception_model.compile(loss='categorical_crossentropy',
                        optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                        metrics=['accuracy']
                       )


# ### DataGenerators

# #### Training DataGenerator

# In[ ]:


# Implement train ImageDataGenerator and specify some preprocessing
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    width_shift_range=0.1,
    height_shift_range=0.1
)


# In[ ]:


# Upload and peprocess images
train_generator = train_datagen.flow(
        X_train, y_train, 
        batch_size=batch_size,
        shuffle=False)  


# #### Validation DataGenerator

# In[ ]:


# Implement validation ImageDataGenerator
validation_datagen = ImageDataGenerator(
    rescale=1./255
)


# In[ ]:


validation_generator = validation_datagen.flow(
        X_val, y_val,
        batch_size=batch_size,
        shuffle=False) 


# #### Test DataGenerator

# In[ ]:


test_datagen = ImageDataGenerator(
    rescale=1./255
)


# In[ ]:


test_generator = test_datagen.flow(
        X_test, y_test,
        batch_size=batch_size,
        shuffle=False
)


# ### Callbacks

# Before we start training our model we should care about avoiding of model overfitting. Callback functions will be very helpfull.  
# 
# > A callback is a set of functions to be applied at given stages of the training procedure. You can use callbacks to get a view on internal states and statistics of the model during training.
# 
# There are two useful ones: **Early Stop** (Keras EarlyStop() method) and **Learning Rate Reduction** (Keras ReduceLROnPlateau() method). 
# 
# **Nota Bene**  
# In Kaggle enviroment we should import EarlyStop and ReduceLROnPlateau from tensorflow.keras.callbacks, not from keras.callbacks, since the last doesn't accepted in model.fit().

# #### Early Stop

# In[ ]:


# Stop model learning after 10 epochs in which val_loss value not decreased
early_stop = EarlyStopping(patience=10, 
                          verbose=1, 
                          mode='auto'
                         )


# #### Learning Rate Reduction

# In[ ]:


# Reduce the learning rate when accuracy, for example, not increase for two continuous steps
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001
                                           )


# In[ ]:


# Save callbacks
callbacks = [early_stop, learning_rate_reduction]
callbacks


# ### Model fit

# In[ ]:


hist = inception_model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_X_val//batch_size,
    steps_per_epoch=total_X_train//batch_size,
    callbacks=callbacks
)


# In[ ]:


# poch 1/20
# 432/432 [==============================] - 25s 57ms/step - loss: 0.9134 - accuracy: 0.6843 - val_loss: 0.7519 - val_accuracy: 0.7616 - lr: 1.0000e-04
# Epoch 2/20
# 432/432 [==============================] - 23s 53ms/step - loss: 0.6030 - accuracy: 0.7841 - val_loss: 0.6113 - val_accuracy: 0.7801 - lr: 1.0000e-04
# Epoch 3/20
# 432/432 [==============================] - 23s 54ms/step - loss: 0.5084 - accuracy: 0.8154 - val_loss: 0.5370 - val_accuracy: 0.8079 - lr: 1.0000e-04
# Epoch 4/20
# 432/432 [==============================] - 22s 52ms/step - loss: 0.4568 - accuracy: 0.8336 - val_loss: 0.5346 - val_accuracy: 0.8009 - lr: 1.0000e-04
# Epoch 5/20
# 432/432 [==============================] - 24s 54ms/step - loss: 0.4375 - accuracy: 0.8435 - val_loss: 0.5041 - val_accuracy: 0.8148 - lr: 1.0000e-04
# Epoch 6/20
# 432/432 [==============================] - 22s 52ms/step - loss: 0.3996 - accuracy: 0.8528 - val_loss: 0.4885 - val_accuracy: 0.8218 - lr: 1.0000e-04
# Epoch 7/20
# 432/432 [==============================] - 24s 54ms/step - loss: 0.3680 - accuracy: 0.8620 - val_loss: 0.5028 - val_accuracy: 0.8194 - lr: 1.0000e-04
# Epoch 8/20
# 432/432 [==============================] - ETA: 0s - loss: 0.3690 - accuracy: 0.8667
# Epoch 00008: ReduceLROnPlateau reducing learning rate to 4.999999873689376e-05.
# 432/432 [==============================] - 25s 57ms/step - loss: 0.3690 - accuracy: 0.8667 - val_loss: 0.5066 - val_accuracy: 0.8194 - lr: 1.0000e-04
# Epoch 9/20
# 432/432 [==============================] - 23s 52ms/step - loss: 0.3142 - accuracy: 0.8945 - val_loss: 0.4857 - val_accuracy: 0.8287 - lr: 5.0000e-05
# Epoch 10/20
# 432/432 [==============================] - 24s 55ms/step - loss: 0.2965 - accuracy: 0.8867 - val_loss: 0.4750 - val_accuracy: 0.8380 - lr: 5.0000e-05
# Epoch 11/20
# 432/432 [==============================] - 23s 54ms/step - loss: 0.2845 - accuracy: 0.9043 - val_loss: 0.4753 - val_accuracy: 0.8403 - lr: 5.0000e-05
# Epoch 12/20
# 432/432 [==============================] - 23s 53ms/step - loss: 0.2874 - accuracy: 0.8939 - val_loss: 0.4844 - val_accuracy: 0.8310 - lr: 5.0000e-05
# Epoch 13/20
# 431/432 [============================>.] - ETA: 0s - loss: 0.2748 - accuracy: 0.9030
# Epoch 00013: ReduceLROnPlateau reducing learning rate to 2.499999936844688e-05.
# 432/432 [==============================] - 25s 57ms/step - loss: 0.2744 - accuracy: 0.9032 - val_loss: 0.4928 - val_accuracy: 0.8356 - lr: 5.0000e-05
# Epoch 14/20
# 432/432 [==============================] - 23s 54ms/step - loss: 0.2689 - accuracy: 0.9058 - val_loss: 0.4824 - val_accuracy: 0.8333 - lr: 2.5000e-05
# Epoch 15/20
# 432/432 [==============================] - ETA: 0s - loss: 0.2675 - accuracy: 0.9064
# Epoch 00015: ReduceLROnPlateau reducing learning rate to 1.249999968422344e-05.
# 432/432 [==============================] - 24s 56ms/step - loss: 0.2675 - accuracy: 0.9064 - val_loss: 0.4757 - val_accuracy: 0.8333 - lr: 2.5000e-05
# Epoch 16/20
# 432/432 [==============================] - 23s 53ms/step - loss: 0.2451 - accuracy: 0.9130 - val_loss: 0.4748 - val_accuracy: 0.8333 - lr: 1.2500e-05
# Epoch 17/20
# 432/432 [==============================] - ETA: 0s - loss: 0.2615 - accuracy: 0.9145
# Epoch 00017: ReduceLROnPlateau reducing learning rate to 1e-05.
# 432/432 [==============================] - 23s 53ms/step - loss: 0.2615 - accuracy: 0.9145 - val_loss: 0.4769 - val_accuracy: 0.8287 - lr: 1.2500e-05
# Epoch 18/20
# 432/432 [==============================] - 24s 57ms/step - loss: 0.2516 - accuracy: 0.9107 - val_loss: 0.4685 - val_accuracy: 0.8333 - lr: 1.0000e-05
# Epoch 19/20
# 432/432 [==============================] - 24s 55ms/step - loss: 0.2439 - accuracy: 0.9186 - val_loss: 0.4724 - val_accuracy: 0.8333 - lr: 1.0000e-05
# Epoch 20/20
# 432/432 [==============================] - 24s 55ms/step - loss: 0.2454 - accuracy: 0.9145 - val_loss: 0.4674 - val_accuracy: 0.8426 - lr: 1.0000e-05


# We obtain accuracy 90 %. Good result. Howewer, it can be improved by playing with model hyperpaameters.

# ### Visualize accuracy and loss after model fit

# In[ ]:


# Plot accuracy and loss curves
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7))

ax1.plot(hist.history['loss'], color='r', label="Train loss")
ax1.plot(hist.history['val_loss'], color='b', label="Validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
legend = ax1.legend(loc='best', shadow=True)

ax2.plot(hist.history['accuracy'], color='r', label="Train accuracy")
ax2.plot(hist.history['val_accuracy'], color='b',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))
legend = ax2.legend(loc='best', shadow=True)

plt.tight_layout()
plt.show()


# ### Predict on validation data

# In[ ]:


# Predict on validation X_val_resnet
y_pred_val = inception_model.predict_generator(validation_generator)


# In[ ]:


# Prepare y_true and y_pred on validation by taking the most likely class
y_true_val = y_val.argmax(axis=1)
y_pred_val = y_pred_val.argmax(axis=1)


# In[ ]:


# Check datatypes
print(f'y_true datatype: {y_true_val.dtype}')
print(f'y_pred datatype: {y_pred_val.dtype}')


# In[ ]:


# Evaluate on validation dataset
loss, acc = inception_model.evaluate_generator(validation_generator, verbose=0)
print(f'Validation loss: {loss:.2f}%')
print(f'Validation accuracy: {acc*100:.2f}%')


# ### Visualize prediction on validation data

# In[ ]:


# Compute and plot the Confusion matrix
confusion_mtx_resnet = confusion_matrix(y_true_val, y_pred_val) 

f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx_resnet, annot=True, fmt='d', cmap=plt.cm.Blues)
plt.xlabel("Predicted Label")
plt.ylabel("Validation (aka True) Label")
plt.title("Confusion Matrix")
plt.show()


# ### Predict on test data

# In[ ]:


samples = total_X_test


# In[ ]:


predict = inception_model.predict_generator(test_generator, steps=np.ceil(samples/batch_size))


# In[ ]:


predict.shape


# In[ ]:


X_test.shape


# In[ ]:


# Evaluate on test dataset
loss, acc = inception_model.evaluate_generator(test_generator, verbose=0)
print(f'Test loss: {loss:.2f}%')
print(f'Test accuracy: {acc*100:.2f}%')


# In[ ]:


# Get most likely class as y_pred and y_test
y_pred = predict.argmax(axis=1)
y_true = y_test.argmax(axis=1)


# In[ ]:


# Show classification report
print(metrics.classification_report(y_true, y_pred))


# In[ ]:


# Compute and plot the Confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred) 

f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True)
plt.xlabel("Predicted Label")
plt.ylabel("Validation (aka True) Label")
plt.title("Confusion Matrix")
plt.show()


# # Conclusion

# We solved a flower classification problem using Machine Learning method - a CNN (Inception v3 type) using Transfer Learning approach with 90% accuracy. During Machine Learning process we devided flowers dataset into three parts: train, validation and test. First shown model validation data and then made final prediction on test. This is a classical approach. Howewer, accuaracy can be increased using more epochs or tuning model's architecture or/and its hyperparameters.

# # References
# 

# [1] https://arxiv.org/abs/1512.00567  
# [2] https://machinelearningmastery.com/how-to-improve-performance-with-transfer-learning-for-deep-learning-neural-networks/  
# [3] https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/
