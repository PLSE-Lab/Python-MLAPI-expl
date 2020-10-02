#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# This is a 5 layers Sequential Convolutional Neural Network for digits recognition trained on MNIST dataset. I choosed to build it with keras API (Tensorflow backend) which is very intuitive. Firstly, I will prepare the data (handwritten digits images) then i will focus on the CNN modeling and evaluation.

# ### Importing the Libraries and Packages

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools


# In[ ]:


from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.layers import Conv2D
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# In[ ]:


sns.set(style='white', context='notebook', palette='deep')


# ### Import the Data

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.head()


# In[ ]:


y_train = train['label']

# Drop the Label column from train dataset so that we get independent variable in X
X_train = train.drop(['label'], axis = 1 )

# free some space
del train

tr = sns.countplot(y_train)
tr


# In[ ]:


y_train.value_counts()   # Decreasing Order


# ### Let's check the Missing Data

# In[ ]:


X_train.isnull().sum().describe()


# In[ ]:


X_train.isnull().any().describe()    # No missing Value found


# In[ ]:


test.isnull().any().describe()    # No missing Value found


# In our data set we don't find any missing values. So, we can proceed further unhesitantly.
# 
# ### Normalization
# We perform a grayscale normalization to reduce the effect of illumination's differences. Moreover the CNN converg faster on [0..1] data than on [0..255].

# In[ ]:


X_train = X_train/255
test = test/255


# ### Reshape the image

# In[ ]:


X_train = X_train.values.reshape(-1,28,28,1)

#The -1 can be thought of as a flexible value for the library to fill in for you. 
# The restriction here would be that the inner-most shape of the Tensor should be (28, 28, 1). 
# Beyond that, the library can adjust things as needed. In this case, that would be the # of examples in a batch.

test = test.values.reshape(-1,28,28,1)


# ### Label Encoding
# Label encoding is simply converting each value in a column to a number

# In[ ]:


y_train = to_categorical(y_train,num_classes=10)   #from keras.utils.np_utils import to_categorical


# Labels are 10 digits numbers from 0 to 9. We need to encode these lables to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0]).

# ### Split the Training and Validation Set

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size = 0.2, random_state = 2)


# Let's see some images from train data

# In[ ]:


tr = plt.imshow(X_train[0][ :, : , 0] )


# In[ ]:


tr = plt.imshow(X_train[98][ :, : , 0] )


# # CNN
# I am using the Keras Sequential API, where you have just to add one layer at a time, starting from the input.
# 
# The first is the convolutional (Conv2D) layer. It is like a set of learnable filters. I choosed to set 32 filters for the two firsts conv2D layers and 64 filters for the two last ones. Each filter transforms a part of the image (defined by the kernel size) using the kernel filter. The kernel filter matrix is applied on the whole image. Filters can be seen as a transformation of the image.
# 
# The CNN can isolate features that are useful everywhere from these transformed images (feature maps).
# 
# The second important layer in CNN is the pooling (MaxPool2D) layer. This layer simply acts as a downsampling filter. It looks at the 2 neighboring pixels and picks the maximal value. These are used to reduce computational cost, and to some extent also reduce overfitting. We have to choose the pooling size (i.e the area size pooled each time) more the pooling dimension is high, more the downsampling is important.
# 
# Combining convolutional and pooling layers, CNN are able to combine local features and learn more global features of the image.
# 
# Dropout is a regularization method, where a proportion of nodes in the layer are randomly ignored (setting their wieghts to zero) for each training sample. This drops randomly a propotion of the network and forces the network to learn features in a distributed way. This technique also improves generalization and reduces the overfitting.
# 
# 'relu' is the rectifier (activation function max(0,x). The rectifier activation function is used to add non linearity to the network.
# 
# The Flatten layer is use to convert the final feature maps into a one single 1D vector. This flattening step is needed so that you can make use of fully connected layers after some convolutional/maxpool layers. It combines all the found local features of the previous convolutional layers.
# 
# In the end i used the features in two fully-connected (Dense) layers which is just artificial an neural networks (ANN) classifier. In the last layer(Dense(10,activation="softmax")) the net outputs distribution of probability of each class.
# 
# ## Set the CNN model
# #### CNN architechture is In -> [[Conv2D->relu] 2 -> MaxPool2D -> Dropout] 2 -> Flatten -> Dense -> Dropout -> Output

# In[ ]:


obj = Sequential()

obj.add(Conv2D(filters=32,kernel_size=(5,5),padding='same',activation='relu',input_shape = (28,28,1)))
obj.add(Conv2D(filters=32,kernel_size=(5,5),padding='same',activation='relu'))
        
obj.add(MaxPool2D(pool_size=(2,2))) 
obj.add(Dropout(0.25))

obj.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
obj.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
obj.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
obj.add(Dropout(0.25))

obj.add(Flatten())
obj.add(Dense(256, activation = "relu"))
obj.add(Dropout(0.5))
obj.add(Dense(10, activation = "softmax"))


# ### Set the optimizer and annealer
# Once our layers are added to the model, we need to set up a score function, a loss function and an optimisation algorithm.
# 
# We define the loss function to measure how poorly our model performs on images with known labels. It is the error rate between the oberved labels and the predicted ones. We use a specific form for categorical classifications (>2 classes) called the "categorical_crossentropy".
# 
# The most important function is the optimizer. This function will iteratively improve parameters (filters kernel values, weights and bias of neurons ...) in order to minimise the loss.
# 
# I choosed RMSprop (with default values), it is a very effective optimizer. The RMSProp update adjusts the Adagrad method in a very simple way in an attempt to reduce its aggressive, monotonically decreasing learning rate. We could also have used Stochastic Gradient Descent ('sgd') optimizer, but it is slower than RMSprop.
# 
# The metric function "accuracy" is used is to evaluate the performance our model. This metric function is similar to the loss function, except that the results from the metric evaluation are not used when training the model (only for evaluation).

# In[ ]:


optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# In[ ]:


# Compile the model
obj.compile(optimizer=optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])


# In order to make the optimizer converge faster and closest to the global minimum of the loss function, i used an annealing method of the learning rate (LR).
# 
# The LR is the step by which the optimizer walks through the 'loss landscape'. The higher LR, the bigger are the steps and the quicker is the convergence. However the sampling is very poor with an high LR and the optimizer could probably fall into a local minima.
# 
# Its better to have a decreasing learning rate during the training to reach efficiently the global minimum of the loss function.
# 
# To keep the advantage of the fast computation time with a high LR, i decreased the LR dynamically every X steps (epochs) depending if it is necessary (when accuracy is not improved).
# 
# With the ReduceLROnPlateau function from Keras.callbacks, i choose to reduce the LR by half if the accuracy is not improved after 3 epochs.

# In[ ]:


# Set a learning rate annealer
learning_rate = ReduceLROnPlateau(monitor='val_acc', patience=3,verbose=1,factor=0.5,min_lr=0.00001)

epochs = 1
batch_size = 86


# ### Data augmentation
# In order to avoid overfitting problem, we need to expand artificially our handwritten digit dataset. We can make your existing dataset even larger. The idea is to alter the training data with small transformations to reproduce the variations occuring when someone is writing a digit.
# 
# For example, the number is not centered The scale is not the same (some who write with big/small numbers) The image is rotated...
# 
# Approaches that alter the training data in ways that change the array representation while keeping the label the same are known as data augmentation techniques. Some popular augmentations people use are grayscales, horizontal flips, vertical flips, random crops, color jitters, translations, rotations, and much more.
# 
# By applying just a couple of these transformations to our training data, we can easily double or triple the number of training examples and create a very robust model.
# 
# The improvement is important :
# 
# Without data augmentation i obtained an accuracy of 98.11%
# 
# With data augmentation i achieved 99.67% of accuracy (When you increase the no of epochs for example 25)
# 
# Note :- When putting epochs = 25 it may takes several hours.

# In[ ]:


# With data augmentation to prevent overfitting (accuracy 0.99286)

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


# For the data augmentation, i choosed to :
# 
# Randomly rotate some training images by 10 degrees Randomly Zoom by 10% some training images Randomly shift images horizontally by 10% of the width Randomly shift images vertically by 10% of the height I did not apply a vertical_flip nor horizontal_flip since it could have lead to misclassify symetrical numbers such as 6 and 9.
# 
# Once our model is ready, we fit the training dataset .

# In[ ]:


history = obj.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate])


# In[ ]:


# Without Data Augmentation
hist = obj.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, 
         validation_data = (X_val, y_val), verbose = 2)


# ### Confusion Matrix

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
y_pred = obj.predict(X_val)
# Convert predictions classes to one hot vectors 
y_pred_classes = np.argmax(y_pred,axis = 1) 
# Convert validation observations to one hot vectors
y_true = np.argmax(y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10))


# Here we can see that our CNN performs very well on all digits with few errors considering the size of the validation set (4 200 images).
# 
# However, it seems that our CNN has some little troubles with the 4 digits, hey are misclassified as 9. Sometime it is very difficult to catch the difference between 4 and 9 when curves are smooth.
# 
# Let's investigate for errors.
# 
# I want to see the most important errors . For that purpose i need to get the difference between the probabilities of real value and the predicted ones in the results.

# In[ ]:


# Display some error results 

# Errors are difference between predicted labels and true labels
errors = (y_pred_classes - y_true != 0)

y_pred_classes_errors = y_pred_classes[errors]
y_pred_errors = y_pred[errors]
y_true_errors = y_true[errors]
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
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

# Probabilities of the wrong predicted numbers
y_pred_errors_prob = np.max(y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, y_pred_classes_errors, y_true_errors)


# In[ ]:


# predict results
results = obj.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")


# In[ ]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)


# # Thank you.
# 
