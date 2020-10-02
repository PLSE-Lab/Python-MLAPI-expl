#!/usr/bin/env python
# coding: utf-8

# <img src="https://en.mlab.ai/sites/default/files/inline-images/handwritten_numbers.png">

# The objective of this notebook is to create a model on TPUs that allows to correctly classify a handwritten number. The TPUs will allow to distribute the calculations during the learning of the model.

# # <div id="summary">Summary</div>
# 
# **<font size="2"><a href="#chap1">1. Load libraries and check TPU settings</a></font>**
# **<br><font size="2"><a href="#chap2">2. EDA and preprocessing</a></font>**
# **<br><font size="2"><a href="#chap3">3. CNN</a></font>**
# **<br><font size="2"><a href="#chap4">4. Evaluation</a></font>**

# # <div id="chap1">1. Load libraries and check TPU settings</div>

# In[ ]:


# Remove warning messages
import warnings
warnings.filterwarnings('ignore')

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly
import plotly.graph_objects as go
get_ipython().run_line_magic('matplotlib', 'inline')

import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

import tensorflow as tf
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import top_k_categorical_accuracy, categorical_accuracy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


# In[ ]:


# Set seed
np.random.seed(42)


# In[ ]:


print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE


# In[ ]:


# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[ ]:


PATH_TO_DATA = '../input/digit-recognizer/'


# In[ ]:


# Load train and test
train = pd.read_csv(PATH_TO_DATA + 'train.csv')
test = pd.read_csv(PATH_TO_DATA + 'test.csv')


# In[ ]:


# First rows of train
train.head()


# --------
# 
# **<font size="2"><a href="#summary">Back to summary</a></font>**

# # <div id="chap2">2. EDA and preprocessing</div>

# ## <font color='blue'> 2.1 Class distribution</font>

# In[ ]:


def plot_distribution_classes(x_values, y_values):

    fig = go.Figure(data=[go.Bar(
                x=x_values, 
                y=y_values,
                text=y_values
    )])

    fig.update_layout(height=600, width=1200, title_text="Distribution of classes")
    fig.update_xaxes(type="category")

    fig.show()


# In[ ]:


x = np.sort(train.label.unique())
y = train.label.value_counts().sort_index()

plot_distribution_classes(x, y)


# ## <font color='blue'>2.2 Preprocessing</font>

# In[ ]:


def preprocessing(train, test, split_train_size = 0.1):

    X_train = train.drop(["label"],
                         axis = 1)
    y_train = train["label"]

    # Normalize the data
    X_train = X_train / 255.0
    test = test / 255.0

    # Reshape into right format vectors
    X_train = X_train.values.reshape(-1,28,28,1)
    X_test = test.values.reshape(-1,28,28,1)

    # Apply ohe on labels
    y_train = to_categorical(y_train, num_classes = 10)
    
    # Split the train and the validation set for the fitting
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = split_train_size, random_state=42)
    
    return X_train, y_train, X_val, y_val, X_test

X_train, y_train, X_val, y_val, X_test = preprocessing(train, test)


# In[ ]:


print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)


# ## <font color='blue'>2.3 Display some examples</font>

# In[ ]:


# Function that prints images taken randomly of each class with label TODO
def display_images(graph_indexes = np.arange(9)):
    
    # plot first few images
    plt.figure(figsize=(12,12))
    
    for graph_index in graph_indexes:
        
        index = random.randint(1, X_train.shape[0])
        
        # Get corresponding label
        label = list(y_train[index]).index(1)
        
        # define subplot
        plt.subplot(330 + 1 + graph_index)
        plt.title('Label: %s \n'%label,
                 fontsize=18)
        # plot raw pixel data
        plt.imshow(X_train[index][:,:,0], cmap=plt.get_cmap('gray'))
        
    plt.subplots_adjust(bottom = 0.001)  # the bottom of the subplots of the figure
    plt.subplots_adjust(top = 0.99)
    # show the figure
    plt.show()


# In[ ]:


display_images()


# ## <font color='blue'>2.4 Convert data to a tensorflow dataset</font>

# In[ ]:


batch_size = 32 * strategy.num_replicas_in_sync # this is 8 on TPU v3-8, it is 1 on CPU and GPU


# In[ ]:


# Put data in a tensor format for parallelization

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_train.astype(np.float32), y_train.astype(np.float32)))
    .repeat()
    .shuffle(2048)
    .batch(batch_size)
    .prefetch(AUTO)
)

val_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_val.astype(np.float32), y_val.astype(np.float32)))
    .batch(batch_size)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(X_test.astype(np.float32))
    .batch(batch_size)
)


# **<font size="2"><a href="#summary">Back to summary</a></font>**
# 
# -------

# # <div id="chap3">3. CNN</div>

# ## <font color='blue'>3.1 What is a CNN ?</font>
# 
# A CNN is quite similar to Classic Neural Networks (RegularNets) where there are neurons with weights and biases. Just like in RegularNets, we use a loss function and an optimizer in CNNs. Additionally though, in CNNs, there are Convolutional Layers, Pooling Layers, and Flatten Layers. CNNs are mainly used for image classification.
# 
# ### CNN layers
# * **Convolutional layer** 
# 
# The very first layer where we extract features from the images in our datasets. Due to the fact that pixels are only related with the adjacent and close pixels, convolution allows us to preserve the relationship between different parts of an image. Convolution is basically filtering the image with a smaller pixel filter to decrease the size of the image without loosing the relationship between pixels. When we apply convolution to 5x5 image by using a 3x3 filter with 1x1 stride (1 pixel shift at each step). We will end up having a 3x3 output (64% decrease in complexity).
# 
# 
# * **Pooling layer**
# 
# When constructing CNNs, it is common to insert pooling layers after each convolution layer to reduce the spatial size of the representation to reduce the parameter counts which reduces the computational complexity. In addition, pooling layers also **helps with the overfitting problem**. Basically we select a pooling size to reduce the amount of the parameters by selecting the maximum, average, or sum values inside these pixels.
# 
# 
# * **Flatten layer**
# 
# Flattens the input. Does not affect the batch size.

# In[ ]:


# Parameters
epochs = 100
n_steps = X_train.shape[0]//batch_size


# In[ ]:


# Define a custom metric
def top_5_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


# In[ ]:


def CNN_model():
    
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', 
                     activation ='relu', input_shape = (28,28,1)))
    model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', 
                     activation ='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', 
                     activation ='relu'))
    model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', 
                     activation ='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(tf.keras.layers.Dropout(0.25))


    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation = "relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation = "softmax"))
    
    return model


# ## <font color='blue'>3.2 Create model with TPU</font>

# In[ ]:


# TPU
with strategy.scope():
    model = CNN_model()
    
model.summary()

# Compile the model
model.compile(optimizer = 'Adam', 
              loss = "categorical_crossentropy", 
              metrics=["accuracy", top_5_categorical_accuracy])


# ## <font color='blue'>3.3 Good reflexes to have</font>
# 
# * **Add dropout**
# 
# Dropout refers to ignoring neurons during the training phase of certain set of neurons which is chosen at random.
# 
# * **LeakyRelu**
# 
# The advantage of using Leaky ReLU instead of ReLU is that in this way we cannot have vanishing gradient. Parametric ReLU has the same advantage with the only difference that the slope of the output for negative inputs is a learnable parameter while in the Leaky ReLU it's a hyperparameter.
# 
# * **Add callbacks**
# 
# A callback is a function that is to be executed after another function has finished executing hence the name 'call back'. With callbacks, you can define earlystopping criterias for your model if it doesn't learn anymore through epochs. Callback allows you to store some information at the end of each epoch so you can check your model's performance.

# In[ ]:


# Define callbacks

# Save weights only for best model
checkpointer = ModelCheckpoint(filepath = 'weights_best_MNIST.hdf5', 
                               verbose = 2, 
                               save_best_only = True)

# LR strategy
learning_rate = ReduceLROnPlateau(monitor='accuracy', 
                                  patience=7, 
                                  verbose=2, 
                                  factor=0.75)

# If score doesn't improve during patience=20 epochs, stop learning
estopping = EarlyStopping(monitor='val_loss', 
                          patience=14, 
                          verbose=2)


# In[ ]:


history = model.fit(train_dataset, 
                    steps_per_epoch = n_steps, 
                    epochs = epochs, 
                    validation_data=(val_dataset),
                    callbacks = [checkpointer, learning_rate, estopping])


# ## <font color='blue'>3.4 History of CNN</font>

# In[ ]:


def plot_history(model_history):

    plt.figure(figsize = (20,15))
    
    plt.subplot(221)
    # summarize history for accuracy
    plt.plot(model_history.history['top_5_categorical_accuracy'])
    plt.plot(model_history.history['val_top_5_categorical_accuracy'])
    plt.title('top_3_categorical_accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()
    
    plt.subplot(222)
    # summarize history for accuracy
    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()
    
    plt.subplot(223)
    # summarize history for loss
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()
    
    plt.subplot(224)
    # summarize history for lr
    plt.plot(model_history.history['lr'])
    plt.title('learning rate')
    plt.ylabel('lr')
    plt.xlabel('epoch')
    plt.grid()
    
    plt.show()


# In[ ]:


plot_history(history)


# **<font size="2"><a href="#summary">Back to summary</a></font>**
# 
# --------

# # <div id="chap4">4. Evaluation</div>

# In[ ]:


# TPU
with strategy.scope():
    # loading the model with the best validation accuracy
    model.load_weights('weights_best_MNIST.hdf5')
    
model.evaluate(val_dataset)


# ## <font color='blue'>4.1 Confusion Matrix</font>

# In[ ]:


def plot_confusion_matrix(confusion_matrix, 
                          cmap=plt.cm.Reds):
    
    classes = range(10)
    
    plt.figure(figsize=(8,8))
    plt.imshow(confusion_matrix, 
               interpolation='nearest', 
               cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, confusion_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


# Predict the values from the validation dataset
y_pred = model.predict(X_val.astype(np.float32))
# Convert predictions classes to one hot vectors 
y_pred_classes = np.argmax(y_pred, axis = 1) 
# Convert validation observations to one hot vectors
y_true = np.argmax(y_val, axis = 1) 
# compute the confusion matrix
cm = confusion_matrix(y_true, y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(cm)


# ## <font color='blue'>4.2 Predicted images</font>

# In[ ]:


def display_predicted_images(graph_indexes = np.arange(9)):
    
    # plot first few images
    plt.figure(figsize=(12,12))
    
    for graph_index in graph_indexes:
        
        index = random.randint(1, X_val.shape[0])
        
        # Get corresponding label
        predicted_label = y_pred_classes[index]
        true_label = y_true[index]
        
        
        # define subplot
        plt.subplot(330 + 1 + graph_index)
        plt.title('Predicted label: %s \n'%predicted_label+                  'True label %s \n'%true_label,
                 fontsize=18)
        # plot raw pixel data
        plt.imshow(X_val[index][:,:,0], cmap=plt.get_cmap('gray'))
        
    plt.subplots_adjust(bottom = 0.001)  # the bottom of the subplots of the figure
    plt.subplots_adjust(top = 0.99)
    # show the figure
    plt.show()


# In[ ]:


display_predicted_images()


# It is most likely to see a perfect prediction on this short sample of predictions, let's print predictions that have been the least accurate for our model

# ## <font color='blue'>4.3 Errors</font>

# In[ ]:


# Display errors 
errors = (y_pred_classes - y_true != 0)

y_pred_classes_errors = y_pred_classes[errors]
y_pred_errors = y_pred[errors]
y_true_errors = y_true[errors]
X_val_errors = X_val[errors]


# In[ ]:


def display_top9_wrongly_predicted_images(list_of_indexes, graph_indexes = np.arange(9)):
    
    # plot first few images
    plt.figure(figsize=(12,12))
    
    for graph_index in graph_indexes:
        
        index = list_of_indexes[graph_index]
        
        # Get corresponding label
        predicted_label = y_pred_classes_errors[index]
        true_label = y_true_errors[index]
        
        
        # define subplot
        plt.subplot(330 + 1 + graph_index)
        plt.title('Predicted label: %s \n'%predicted_label+                  'True label %s \n'%true_label,
                 fontsize=18)
        # plot raw pixel data
        plt.imshow(X_val_errors[index][:,:,0], cmap=plt.get_cmap('gray'))
        
    plt.subplots_adjust(bottom = 0.001)  # the bottom of the subplots of the figure
    plt.subplots_adjust(top = 0.99)
    # show the figure
    plt.show()


# In[ ]:


# Probabilities of the wrong predicted numbers
y_pred_errors_prob = np.max(y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 9 errors 
most_important_errors = sorted_dela_errors[-9:]

# Show the top 9 errors
display_top9_wrongly_predicted_images(list_of_indexes = most_important_errors)


# **<font size="2"><a href="#summary">Back to summary</a></font>**
# 
# --------

# # Submission

# In[ ]:


# predict results
y_test_pred = model.predict(test_dataset)

# Associate max probability obs with label class
y_test_pred = np.argmax(y_test_pred, axis = 1)
y_test_pred = pd.Series(y_test_pred, name="Label")

submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"), y_test_pred], axis = 1)

submission.to_csv("CNN_model_TPU_submission.csv", index = False)


# # References
# 
# * Thanks to <a href="https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6">yassineghouzam</a> for his inspiring notebook
# 
# * <a href="https://codelabs.developers.google.com/codelabs/keras-flowers-tpu/#2">TPU usage on flowers classification</a>
# 
# * <a href="https://blog.tensorflow.org/2019/01/keras-on-tpus-in-colab.html">TPU documentation</a>
# 
# * My previous notebook: <a href="https://www.kaggle.com/bryanb/handwritten-letters-classification">CNN for Handwritten Letters Classification</a>

# **<font color="red" size="4">Thank you for taking the time to read this notebook. I hope that I was able to answer your questions or your curiosity and that it was quite understandable. If you liked this text, <u>please upvote it</u>. I will really appreciate and this will motivate me to make more and better content !</font>**
