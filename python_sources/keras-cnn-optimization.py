#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# 
# MNIST digit recognition is the "hello world" of image classification. It is a dataset of handwritten digits taken mostly from United States Census Bureau employees. The dataset is a collection of images of digits 0-9, in grayscale and exactly identical dimensions, 28pixels x 28pixels. Pixel values of these images are provided in csv format. Digit recognition is simpler than other compler image recognition problems, but is ideal for  machine learning experiments and evaluating different models.
# 
# In this notebook I explore Convolutional Neural Networks using Keras. Specifically, a couple of techniques used in the process of learning parameters and finding good hyperparameters-
# 
# - **Learning Rate Annealing** - Adapt the learning rate across epochs based on cross validation.
# - **Image Data Augmentation** - Artificially augmenting training samples to reduce overfitting.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
import numpy as np
from numpy import random

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, Callback
from keras import regularizers
from keras.optimizers import Adam


## visualize model using GraphViz
#import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
#from keras.utils import plot_model

def display_images(X, y=[], rows=5, columns=5, cmap="gray"):
    """ Display images and labels
    """
    fig, ax = plt.subplots(rows,columns, figsize=(6,6))
    for row in range(rows):
        for column in range(columns):
            ax[row][column].imshow(X[(row*columns)+column].reshape(28,28), cmap=cmap)
            ax[row][column].set_axis_off()
            if len(y):ax[row][column].set_title("{}:{}".format("label",np.argmax(y[(row*columns)+column])))
    fig.tight_layout()

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load, prepare and preview data
# 
# Our data looks like the sample below. It is a csv file with the true classification in the ```label``` column, followed by 784 pixel values (28x28 pixels unrolled). Each pixel takes a value ranging from 0-255. Since these are black and white images, each pixel is represented by a single value (channel) instead of three separate R, G, B values (3 channels) we would expect in a color image.

# In[ ]:


df = pd.read_csv("../input/train.csv")
#df = pd.read_csv("train.csv")
df.sample(1)


# We pick random samples from these 42000 images to create 3 sets - 
# - training (60%): data used to train convnet
# - cross validation (20%): data used to validate performance
# - test (20%): data used to test classification accuracy
# 
# While there is a separate test set available, we are not using that in this notebook, since it is not labeled and cannot be easily evaluated.
# 

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(df.iloc[:,1:].values, df.iloc[:,0].values, test_size = 0.4)
X_cv, X_test, y_cv, y_test = train_test_split(X_val, y_val, test_size = 0.5)
print("X_train:{}\ny_train:{}\n\nX_cv:{}\ny_cv:{}\n\nX_test:{}\ny_test:{}".format(X_train.shape, y_train.shape, X_cv.shape, y_cv.shape, X_test.shape, y_test.shape))


# The data is in an unrolled format, i.e. each sample is a sequence of 784 pixel values. We will convert this using numpy's reshape function to (28x28x1). i.e. an image that is 28 pixels wide and 28 pixels tall, with 1 channel (black and white image).  So for example, the shape of the training set becomes (25200 samples, 28px, 28px, 1ch)
# 
# We change the output class (label) to categorical or one hot format. i.e. instead of a single value 0-9, we convert this to a array of size 10. e.g.
# y = [9] becomes
# y = [0,0,0,0,0,0,0,0,1,0]
# 
# Additionally, we scale all the features (pixel values) from a range of 0-255, to a range of 0-1. This is done by dividing each value in the feature matrix by 255.
# 
# Here are the new shapes of training, cross validation and test data sets.

# In[ ]:


width = 28
height = 28
channels = 1
X_train = X_train.reshape(X_train.shape[0], width, height, channels)
X_cv = X_cv.reshape(X_cv.shape[0], width, height, channels)
X_test = X_test.reshape(X_test.shape[0], width, height, channels)

# convert output classes to one hot representation
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_cv = np_utils.to_categorical(y_cv, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

X_train = X_train.astype('float32')
X_cv = X_cv.astype('float32')
X_test = X_test.astype('float32')

# Scale features (pixel values) from 0-255, to 0-1 
X_train /= 255
X_cv /= 255
X_test /= 255
print("Reshaped:")
print("X_train:{}\ny_train:{}\n\nX_cv:{}\ny_cv:{}\n\nX_test:{}\ny_test:{}".format(X_train.shape, y_train.shape, X_cv.shape, y_cv.shape, X_test.shape, y_test.shape))


# Here is a preview of a few images in the training set.

# In[ ]:


display_images(X_train, y_train)


# In[ ]:


batch_size=84
epochs=5 # Change to 30
verbose=2

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (5,5), padding="same", activation='relu', input_shape=(width, height, channels) ))
    model.add(Conv2D(32, (5,5), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
    model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(384, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    opt = "adam" #Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

def plot_metrics(h, title=""):
    """ Plot training metrics - loss and accuracy, for each epoch, 
        given a training history object
    """
    fig, axes = plt.subplots(1,2, figsize=(10,5))
      
    axes[0].plot(h.history['loss'], color="lightblue", label="Training", lw=2.0)
    axes[0].plot(h.history['val_loss'], color="steelblue", label="Validation", lw=2.0)

    axes[0].set_title("{} (Loss)".format(title))
    axes[0].set_xlabel("Epoch")
    axes[0].set_xticks(np.arange(len(h.history["loss"]), 2))
    axes[0].set_ylabel("Loss")
    
    axes[1].plot(h.history['acc'], color="lightblue", label="Training", lw=2.0)
    axes[1].plot(h.history['val_acc'], color="steelblue", label="Validation", lw=2.0)
    
    axes[1].set_title("{} (Accuracy)".format(title))
    axes[1].set_xlabel("Epoch")
    axes[1].set_xticks(np.arange(len(h.history["acc"]), 2))
    axes[1].set_ylabel("Accuracy")
    

    for axis in axes:
        axis.ticklabel_format(useOffset=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(loc='best', shadow=False)
    fig.tight_layout()
    
def plot_losses(batch_hist, title=""):
    fig, ax1 = plt.subplots()

    ax1.semilogx(batch_hist.losses)
    ax1.set_title("{} (Batch Loss)".format(title))  
    
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    plt.show()


# ## Model Definition
# 
# I use this model as the starting point, built using Keras sequential API. I create 3 separate instances of this model and then compare results with learning rate annealing and image data augmentation. 
# 
# The code to create the model is wrapped into a function, primarily for consistency, to avoid any inadvertent changes to the model between experiments.
# 
# Keras also provides a easy way to generate this diagram from the model (requires GraphViz and pydot). See code below. It doesn't work on Kaggle, though.
# 
# <img src="https://raw.githubusercontent.com/vinayshanbhag/keras-cnn-mnist/master/model.png"/>

# In[ ]:


model0 = create_model()

# Visualize model using GraphViz
#plot_model(model0, show_shapes=True, show_layer_names=False,to_file='model.png')

model0_batch_hist = LossHistory()

model0_metrics = model0.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, 
          validation_data = (X_cv, y_cv), verbose = verbose, shuffle=True, callbacks=[model0_batch_hist])

#model0.save_weights("model0.h5")


# ## Learning Rate Annealing
# 
# Learning rate is the step size in gradient descent. If the step size is too large, the system may oscillate chaotically. On the other hand, if the step size is too small, it may take too long or may settle on a local minimum. 
# 
# We will watch validation accuracy in each epoch, and reduce the learning rate to a third, if it plateaus in 2 consecutive epochs. Keras provides an aptly named, ```ReduceLROnPlateau```, callback to adapt the learning rate based on results from each epoch. Ref: [```ReduceLROnPlateau```](https://keras.io/callbacks/#reducelronplateau) for more options.
# 
# The verbose mode, allows us to see when this actually kicks in.

# In[ ]:


learning_rate_controller = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=verbose, factor=0.3, min_lr=0.00001, epsilon=0.001)


# ## Model with learning rate annealer
# 
# We create a new instance of the same model, but this time, insert a callback to our learning rate control function defined above. Then fit the model to our training data set and collect metrics.

# In[ ]:


model1 = create_model()
model1_batch_hist = LossHistory()
model1_metrics = model1.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, 
          validation_data = (X_cv, y_cv), verbose = verbose, shuffle=True, callbacks=[learning_rate_controller,model1_batch_hist])
#model1.save_weights("model1.h5")


# ## Data Augmentation
# 
# To reduce overfitting and to improve classification accuracy, we can augment the training samples, with random transformations of images in the training set. In Keras, this is done using ```keras.preprocessing.image.ImageDataGenerator``` class. We can apply random transformations such as, zooming, rotation, shifting the image up/down. We will limit rotation to a few degrees, and disable horizontal and vertical flipping, as our dataset of digits is prone to produce ambiguous results with these operations. 
# 
# See [```ImageDataGenerator```](https://keras.io/preprocessing/image/#imagedatagenerator) for lots of other options that are useful for other types of images.

# In[ ]:


idg = ImageDataGenerator(
        rotation_range=10,
        zoom_range = 0.05, 
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=False,
        vertical_flip=False, data_format="channels_last")


# Here are a few images produced by the image data generator.

# In[ ]:


image_data = idg.flow(X_train,y_train, batch_size=25).next()
print("Sample images from ImageDataGenerator:")
display_images(image_data[0], image_data[1])


# ## Model with data augmentation
# 
# We create yet another instance of the model we defined earlier with learning rate annealer. This time instead of fitting it to the training data set, we will instead fit it to the images generated by the ```ImageDataGenerator```. We will collect loss and accuracy metrics for comparison.

# In[ ]:


model2 = create_model()
model2_batch_hist = LossHistory()
model2_metrics = model2.fit_generator(idg.flow(X_train,y_train, batch_size=batch_size),
                    epochs = epochs,
                    steps_per_epoch=X_train.shape[0]//batch_size,
                    validation_data=(X_cv,y_cv),
                    callbacks=[learning_rate_controller,model2_batch_hist],                         
                    verbose = verbose)
#model2.save_weights("model2.h5")


# ## Comparing the loss functions
# 
# A plot of the loss function over all batches, shows the effects of learning rate annealing and data augmentation. The learning rate annealer appears to reduce noise. 
# 
# Images below are from running the same code over 30 epochs. The code here is scaled down due to Kaggle limits.
# 
# <img src="https://raw.githubusercontent.com/vinayshanbhag/keras-cnn-mnist/master/model0-loss.png?a">
# <img src="https://raw.githubusercontent.com/vinayshanbhag/keras-cnn-mnist/master/model1-loss.png?a">
# <img src="https://raw.githubusercontent.com/vinayshanbhag/keras-cnn-mnist/master/model2-loss.png?a">

# In[ ]:


plot_losses(model0_batch_hist, "CNN")
plot_losses(model1_batch_hist, "CNN with Learning Rate Annealer")
plot_losses(model2_batch_hist, "CNN with Augmented Data")


# ## Results
# 
# Keras provides a [```History```](https://keras.io/callbacks/#history) callback that returns loss and accuracy metrics for training and validation sets for each epoch. Plots of the loss and accuracy metrics on training and validation data from the three models, help us see the effects of learning rate annealing and data augmentation.
# 
# - Learning rate annealer, smoothened the loss and accuracy metrics. 
# - Data augmentation reduced overfitting compared to the previous models.
# 
# Again, images embedded here are from running the same code over 30 epochs. The code here is scaled down due to Kaggle limits.
# 
# <img src="https://raw.githubusercontent.com/vinayshanbhag/keras-cnn-mnist/master/model0-metrics.png"/>
# <img src="https://raw.githubusercontent.com/vinayshanbhag/keras-cnn-mnist/master/model1-metrics.png"/>
# <img src="https://raw.githubusercontent.com/vinayshanbhag/keras-cnn-mnist/master/model2-metrics.png"/>

# In[ ]:


plot_metrics(model0_metrics,"Convolutional Neural Network")
plot_metrics(model1_metrics,"CNN with Learning Rate Annealer\n")
plot_metrics(model2_metrics,"CNN with Annealer and Data Augmentation\n")


# ## Classification Accuracy
# 
# Finally, a summary of how the three models performed in terms of training, validation and test accuracy.
# 
# Chart below is from running the same code over 30 epochs. The code here is scaled down due to Kaggle limits.
# 
# <img src="https://raw.githubusercontent.com/vinayshanbhag/keras-cnn-mnist/master/summary.png"/>

# In[ ]:


models = [model0, model1, model2]
metrics = [model0_metrics, model1_metrics, model2_metrics]
names = ["Convolutional Neural Network", "CNN + Learning Rate Annealing", "CNN + LR + Data Augmentation"
         ]
data = []
for i, m in enumerate(zip(names, metrics, models)):
    data.append([m[0], "{:0.2f}".format(m[1].history["acc"][-1]*100), "{:0.2f}".format(m[1].history["val_acc"][-1]*100), "{:0.2f}".format(m[2].evaluate(X_test, y_test, verbose=0)[1]*100)])

results = pd.DataFrame(data, columns=("Model","Training Accuracy","Validation Accuracy", "Test Accuracy"))
from IPython.display import display, HTML
display(HTML(results.to_html(index=False)))
plt.bar(np.arange(len(results["Model"].values)),results["Training Accuracy"].values.astype("float64"), 0.2, color="lightblue")
plt.bar(np.arange(len(results["Model"].values))+0.2,results["Validation Accuracy"].values.astype("float64"), 0.2, color="steelblue")
plt.bar(np.arange(len(results["Model"].values))+0.4,results["Test Accuracy"].values.astype("float64"), 0.2, color="navy")
plt.ylim(97, 100)
plt.xticks(np.arange(len(results["Model"].values))+0.2, ["CNN","CNN+LR", "CNN+LR+Aug"])
plt.legend(["Training","Validation", "Test"],loc=(1,0.5))
g = plt.gca()
g.spines["top"].set_visible(False)
g.spines["right"].set_visible(False)
plt.title("Accuracy")


# #### Work In Progress

# In[ ]:





# In[ ]:




