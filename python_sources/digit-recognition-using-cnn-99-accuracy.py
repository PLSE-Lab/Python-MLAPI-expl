#!/usr/bin/env python
# coding: utf-8

# # MNIST Handwritten Digit Recognizer using Deep CNN
# ![MNIST Dataset](https://miro.medium.com/max/1400/1*26W2Yk3cu2uz_R8BuSb_SA.png)
#    If you got interested in Machine Learning and Data Science and you are looking for a way to learn the concepts of machine learing, you have came to the perfect place to learn and be nutured. Kaggle is perfect for the begginer to get started with machine learning. Handwritten Digit Recognizer using MNIST Dataset is getting started competition for all beginners. I hope my notebook can introduced you to the world of machine learning and future.
# 
#    In this kernel, I have created a Deep Convolutional Neural Network (CNN) model to recognize different handwritten digits and classify them. The dataset used here is actually from [Digit Recognition Competition](https://www.kaggle.com/c/digit-recognizer). Let's get started.
# 
# **Frameworks**
# * [Tensorflow v2](https://tensorflow.org) - A open sourced machine learning framework from Google.
# * [Keras](https://keras.io) - A open sourced neural network library running on top of tensorflow.
# 
# 
# ### Process flow
# 1. [Importing Libraries](#1)
# 2. [Preparing the Dataset](#2)
# 3. [Model Building](#3)
# 4. [Model Fitting](#4)
# 5. [Analyzing the model](#5)
# 6. [Predicting the test data](#6)
# 
# 
# 

# ## 1. Importing the Libraries <a></a>
# Here we are importing the required libraries for the kernel. 

# In[ ]:


# Importing Tensorflow and keras
#Keras is built into TF 2.0

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as img
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

#Setting the Theme of the data visualizer Seaborn
sns.set(style="dark",context="notebook",palette="muted")

#Tensorflow Version
print("TensorFlow Version:   "+tf.version.VERSION)
print("Keras Version:   "+tf.keras.__version__)


# ## 2. Preparing the Dataset
# We have added the data provided from the MNIST Handwritten Digit Recognition competition. Use the *+ Add data* button on the top right corner to do that. Select the competition. The data will be added to the kernel.We can find the dataset in the right panel under Data. You can copy the path of the train and test data from the right panel. 
# 
# read_csv is used to return a pandas DataFrame object from conversion of csv file.
# 
# Then we select the label column and store it in Y_train which is used for the training. X_train contains the pixel values of the respective labelled image.
# 
# We visualize the total number of data of each class using countplot.Then, we check for missing values in the dataset.

# In[ ]:


train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")


# In[ ]:


Y_train = train['label']

#Dropping Label Column
X_train = train.drop(labels=['label'],axis=1)

#free up some space
del train

graph = sns.countplot(Y_train)
 
Y_train.value_counts()


# In[ ]:


#Checking for any null or missing values
X_train.isnull().any().describe()

test.isnull().any().describe()


# ### Normalisation
# Normalisation is done to reduce the scale of the input values. The pixel value ranges from 0 to 255 which specify gradient of gray. The CNN will converge more faster on values 0 to 1 than 0 to 255. So we divide every value by 255 to scale the data from [0..255] to [0..1]. It helps the model to better learning of features by decreasing computational complexities if we have data that scales bigger.

# In[ ]:


X_train = X_train/255
test = test/255


# ### Reshape
# The array of pixel values are reshaped into a (28,28,1) matrix. We are feeding the CNN model with input_shape of 28x28x1 matrix.

# In[ ]:


X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


# ### Label Encoding
# Since the CNN model will give results in a vector of predictions for each classes. The label (numbers) are encoded into hot vector for prediction by the model. So that we can train the CNN with the encoded outputs and the parameters are tuned accordingly

# In[ ]:


Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=10)
#To enable label into hot vector. For Eg.7 -> [0,0,0,0,0,0,0,1,0,0]


# ### Train and Validation Data Split
# We are segmenting the input data for training into two exclusive data namely, Train and Validation data sets. Train data is used to train the model whereas the validation data is used for cross verification of the model's accuracy and how well the model is generalized for the data other than training data. Validation accuracy and loss will tell us the performance of the model for new data and it will show if there is overfitting or underfitting situation while model training. 
# 

# In[ ]:


#Spliting Train and test set
random_seed =2

X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,test_size=0.1,
                                                random_state = random_seed)


# In[ ]:


#Show some example 

g = plt.imshow(X_train[0][:,:,0])


# ## 3. Model Building

# Deep Convolutional Neural Network is a network of artificial neural networks. A model archiecture is the design of the neural networks with which we train the parameters in training process. Here we used LeNet-5 Architecture, it was proposed by Yann LeCun in 1998. Its pretty popular for its minimal structure and easy to train nature. LeNet-5 architecture is suitable for recognition and classification of different classes of objects in small resolution images. 
# 
# To learn more about CNN architectures and current state-of-the-art architecture, take a visit [here](https://medium.com/analytics-vidhya/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5)
# 
# ![Deep CNN Architecture](https://miro.medium.com/max/3744/1*SGPGG7oeSvVlV5sOSQ2iZw.png)

# In[ ]:


#CNN Architecture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> 
                           #Flatten -> Dense -> Dropout -> Out
model = tf.keras.Sequential()

model.add(layers.Conv2D(filters=32, kernel_size=(5,5), padding='Same', 
                       activation=tf.nn.relu, input_shape = (28,28,1)))
model.add(layers.Conv2D(filters=32, kernel_size=(5,5), padding='Same', 
                       activation=tf.nn.relu))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))


model.add(layers.Conv2D(filters=64, kernel_size=(3,3), padding='Same', 
                       activation=tf.nn.relu, input_shape = (28,28,1)))
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), padding='Same', 
                       activation=tf.nn.relu))
model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(256,activation=tf.nn.relu))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(10,activation=tf.nn.softmax))


# ### Optimizers and Annealers
# * Optimizer is the crucial part in a neural network. Optimizer ensure the model reaches the optimium faster. RMSProp optimizer makes the model converge more effectively and faster. It also stricts the model to coverge at global minimum therefore the accuracy of the model will be higher.  
# * Setting the learning rate is very important in a Deep Learning algorithm. Though choosing a good learning rate may yield its goodness, scheduling a reduction in learning rate while training has a great advantage in convergence at global minimum. ReduceLRonPlateau makes the Learning Rate to reduce at a rate by monitoring the learning rate and epoch. It will greatly help the model to achieve maximum accuracy.

# In[ ]:


#Defining Optimizer

optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# In[ ]:


#Compiling Model

model.compile(optimizer = optimizer, loss='categorical_crossentropy', 
             metrics=["accuracy"])


# In[ ]:


#Setting Learning rate annealer

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                           patience=3,
                                           verbose=1,
                                           factor=0.5,
                                           min_lr=0.00001)


# > I have set the epoch to 10 for the purpose of kernel. You can bump it upto 30 to see a larger accuracy. 

# In[ ]:


epochs=30
batch_size = 112


# ### Data Augmentation
# Data Augmentation is the process of creating more dataset for training by manipulating the given images. In Deep learning, the availability of large dataset is very vital for the training. Since we have limited real world training samples, we can use data augementation to create more images for training. Data Augmentation involves zoom, rotating, flip, crop and other image manipulations over the available datasets to create further more data for training. Data Augmentation makes the model to classify more generally.

# In[ ]:


datagen = ImageDataGenerator(
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


datagen.fit(X_train)


# ## 4. Model Fitting
#    Model Fitting or Model Training is where we train our model and evaluate the error parameters. Training process typically take a lot of time when it runs in a CPU. But the training can be speeeded up with the graphics card that have CUDA support. Kaggle have inbuilt limited GPU Support be sure to turn it on when running this cell.

# In[ ]:


if(tf.test.is_built_with_cuda() == True):
    print("CUDA Available.. Just wait a few moments...")
else: 
    print("CUDA not Available.. May the force be with you.")


# In[ ]:


# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])


# ## 5. Analyzing the model
# We can analyse the model using various methods. One of them, is learning graph. Here we plot the losses of both training and validation data in a plot and evaluate it by looking at the trend. For a ideal model, training and validation loss should be low and similar. 

# In[ ]:


# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# ### Confusion Matrix Plotting
# Confusion Matrix is another way of model evaluation. It is used for grphical representation of performance of the model. It shows the performance of Model in predicting every class. Here you can find the model pretty accurately predict the relevant classes.

# In[ ]:


# Look at confusion matrix 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 


# ### Important Error
# Though we may have very high accuracy, any deep learing model cannot correctly predict all the images. So we are viewing the important errors done by our model for perspection.

# In[ ]:


# Display some error results 

# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title(" Predicted :{} True :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)


# ## 6. Predicting the test data
# Finally we are predicting the test dataset for the competition afer done training and performance evaluation. We predict the test data and store it in a csv file for competition submission.

# In[ ]:


# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")


# In[ ]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)


# ## Conclusion
# I hope you all enjoyed by notebook. It's my first try of writing one. I have learnt many awesome new things about Deep Learning and the math concepts behind it. I got incredibly amused by all those concepts and sink in the world of deep learning. 
# 
# ## If you find this notebook useful, please do upvote which will encourage me to learn and do more in the future.
