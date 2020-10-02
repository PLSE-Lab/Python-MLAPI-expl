#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Networks with Keras
# 
# Started with Yassine Ghouzam's kernel, [Introduction to CNN Keras](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6). Yassine, many thanks and great work! I wanted to build on his work by adding some more commentary (so that I could learn, but hopefully you do too!), and of course to make my own tweaks and experiment a little.
#  
# Still need to try:
#  - Ranger optimizer
# 
# ### Table of Contents
# 1. Introduction
# 1. Data preparation
# 1. Model building
# 1. Model training
# 1. Prediction and submission

# # 1. Introduction
# 
# In this notebook we will use an ensemble of sequential convolutional neural networks for digit recognition on the MNIST dataset. Ensembling, through a simple majority vote, allows us to improve our model's accuracy while greatly reducing the variance of the predictions. The Keras framework presents a very intuitive and user-friendly interface with which we can build and customize our neural networks. Training is done via CPU, but can be much faster by taking advantage of GPU resources.
# 
# This Notebook follows three main parts:
# * Data preparation
# * CNN modeling and evaluation
# * Results prediction and submission
# 
# 
# A preview of the MNIST dataset:
# 
# <img src="https://corochann.com/wp-content/uploads/2017/02/mnist_plot.png" ></img>

# In[ ]:


# import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils import plot_model
from IPython.display import Image
import itertools

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

SEED = 999
np.random.seed(SEED)
sns.set(style='white', context='notebook', palette='pastel')


# # 2. Data preparation
# ### 2.1 Load the data

# In[ ]:


X_train = pd.read_csv('../input/train.csv')
X_test = pd.read_csv('../input/test.csv')


# Train and test images (28px x 28px) are now stored as a pandas DataFrame of 1D vectors with length 784 (=$28^2$). We reshape these vectors to 3D matrices with shape 28x28x1. This third dimension of 1 corresponds to the channel -- MNIST images are grayscale so we need only one channel.

# ### 2.2 Organize and examine

# In[ ]:


Y_train = X_train['label']
X_train = X_train.drop(labels=['label'], axis=1) 

print(X_train.isnull().any().describe())

# normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# resize
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)


# In[ ]:


# basic frequency plots
g = sns.countplot(Y_train)
Y_train.value_counts()


# ### 2.3 Preview
# 
# We can get a better sense for one of these examples by visualizing the image and looking at the label:

# In[ ]:


# preview an image
print('Label:', Y_train[0])
g = plt.imshow(X_train[0][:,:,0])


# ### 2.4 Prepare for training
# 
# We split the original `X_train` dataset in two parts: a small fraction (10%) for the validation set and the rest (90%) is used to train the model.
# 
# Since we have 42,000 training images of balanced labels, a random split of the training set doesn't cause some labels to be over represented in the validation set. Be careful with unbalanced datasets as a simple random split could cause inaccurate evaluation during the validation. To avoid that, you could use the `stratify=True` option of `train_test_split`, or look into the ADASYN and SMOTE over-sampling algorithms. 

# In[ ]:


# one-hot encoding
Y_train = to_categorical(Y_train, num_classes=10)

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                  test_size=0.1, random_state=SEED)


# # 3. Model building
# ### 3.1 Define the model

# The Keras Sequential API allows us to add one layer at a time, starting from the input layer. Combining convolutional and pooling layers, CNNs are able to learn local features of the image as well as learn more global context information.
# 
# A Conv2D layer is like a set of learnable filters, where each filter transforms a part of the image (defined by the kernel size) using a kernel filter matrix. This matrix is applied on the whole image and can be seen as a transformation of the image. The CNN can then isolate features that are useful everywhere from these transformed images (feature maps). I chose to set 32 filters for the two first conv2D layers and 64 filters for the two last ones.
# 
# On the other hand, a pooling (MaxPool2D) layer simply acts as a downsampling filter. It looks at the 2 neighboring pixels and picks the maximal value. These are used to reduce computational cost, and to some extent also reduce overfitting. We specify the pooling size (i.e the area size pooled each time) -- the greater the pooling area, the greater the loss of information. Having pooling area and strides of 2x2 is a very common choice.
# 
# Dropout is a regularization method where a proportion of nodes in the layer are randomly ignored (setting their weights to zero) for each training sample. This random dropping of nodes in the network forces it to learn features in a more distributed way, improving generalization and reducing overfitting. 
# 
# `relu`, short for Rectified Linear Unit, is the activation function max(0, x). It is used to add non linearity to the network. 
# 
# The Flatten layer is use to convert the final feature maps into a one single 1D vector. This flattening step is needed so that you can make use of fully connected layers after some convolutional/maxpool layers. It combines all the found local features of the previous convolutional layers.
# 
# An optimizer will iteratively improve parameters (filters kernel values, weights and bias of neurons, etc.) in order to minimize the loss. The Adam optimizer in particular combines the features of RMSProp and momentum, and has proven to be a very popular and efficient option.
# 
# In the end I used the features in two fully-connected (Dense) layers which is just an artificial neural networks (ANN) classifier. In the last layer, `Dense(10, activation='softmax')` uses the softmax activation function to output the probabilities for each class, with `10` indicating the dimension of our output (for the 10 possible digits).
# 
# Because each net's weights are randomly initialized, we can know that adding additional neural networks will have some natural variation in their results and not simply all calculate identical results.

# ### 3.2 Creating an ensemble of models
# 
# Chris Deotte has a very good [kernl](https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist) that makes use of ensembling, and it is both very informative and powerful. I encourage you to check it out if you haven't already. He also has [another one](https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist) with more insights on how to decide on your neural network's architecture. I decided to reduce the number of models from 15 to save training time, and since it seemed a bit excessive to have so many - though, it can obviously achieve some very good results!

# In[ ]:


def create_models(num_nets):
    models = [0] * num_nets

    for i in range(num_nets):
        models[i] = Sequential()

        models[i].add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu',
                     input_shape=(28,28,1)))
        models[i].add(BatchNormalization())
        models[i].add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu'))
        models[i].add(BatchNormalization())
        models[i].add(MaxPool2D(pool_size=(2,2)))
        models[i].add(Dropout(0.25))

        models[i].add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
        models[i].add(BatchNormalization())
        models[i].add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
        models[i].add(BatchNormalization())
        models[i].add(MaxPool2D(pool_size=(2,2)))
        models[i].add(Dropout(0.25))

        models[i].add(Flatten())
        models[i].add(Dense(256, activation='relu'))
        models[i].add(Dropout(0.4))
        models[i].add(Dense(10, activation='softmax'))
    
        models[i].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print('Compiled model {}.'.format(i+1))
    
    return num_nets, models


# In[ ]:


num_nets, models = create_models(num_nets=7)


# **Visualize our neural network**
# 
# Thanks to [Satonio](https://www.kaggle.com/acleon/the-acleon-s-first-competion) for illustrating this - a pretty cool feature!

# In[ ]:


plot_model(models[0], to_file='model.png', show_shapes=True, show_layer_names=True)
Image('model.png')


# ### 3.3 Learning rate annealer
# 
# The learning rate is the step by which the optimizer walks through the 'loss landscape'. The higher LR, the bigger are the steps and the quicker is the convergence. However the sampling is very poor with a high learning rate, and the optimizer could fall into a local minimum.
# 
# In order to make the optimizer converge faster and more closely to the global minimum of the loss function, we can use an *annealer*. To keep the advantage of the faster computation time with a higher learning rate, we can decrease it dynamically every X steps (epochs) but only if necessary (when accuracy is not improved).
# 
# With the ReduceLROnPlateau function from keras.callbacks, we reduce the learning rate by half if the validation set accuracy is not improved after 3 epochs.

# In[ ]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, 
                                            factor=0.5, min_lr=0.00001)


# ### 3.4 Data augmentation 

# In order to avoid overfitting, we need to artificially expand our handwritten digit dataset. The idea is to alter the training data with small transformations to reproduce the variations occuring when someone is writing a digit.
# 
# For example:
# * The number is not centered 
# * The scale is not the same (some who write with big/small numbers)
# * The image is rotated
# 
# Approaches that alter the training data in ways that change the array representation while keeping the label the same are known as data augmentation techniques. Some popular augmentations people use are grayscales, horizontal flips, vertical flips, random crops, color jitters, translations, rotations, and much more. 
# 
# By applying just a couple of these transformations to our training data, we can easily double or triple the number of training examples and create a very robust model.
# 
# For data augmentation, we will:
#    - Randomly rotate some training images by 10 degrees
#    - Randomly zoom by 10% some training images
#    - Randomly shift horizontally by 10% of the total image width
#    - Randomly shift vertically by 10% of the total image height
#    
# The parameters `vertical_flip` and `horizontal_flip` don't make sense to apply here because of the symmetrical relationship of some numbers like 6 and 9 -- we don't want to mix them up.

# In[ ]:


datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.10, 
                             width_shift_range=0.10, height_shift_range=0.10)


# Now that our models are ready, we train:

# # 4. Model training

# In[ ]:


def train_models(models, epochs):
    num_nets = len(models)
    history = [0] * num_nets
    
    for i in range(num_nets):
        X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size=0.1)
        history[i] = models[i].fit_generator(datagen.flow(X_train2, Y_train2, batch_size=64),
                                            epochs=epochs, steps_per_epoch=X_train2.shape[0]//64,  
                                            validation_data=(X_val2,Y_val2),
                                            callbacks=[learning_rate_reduction], verbose=0)
        
        print('CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}'.format(
            i+1, epochs, max(history[i].history['accuracy']), max(history[i].history['val_accuracy'])))
    
    return models, history


# In[ ]:


models, history = train_models(models=models, epochs=18)


# ### 4.1 Evaluation
# 
# **Confusion matrix**

# In[ ]:


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
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Predict the values from the validation dataset 
Y_pred = np.zeros((X_val.shape[0], 10)) 

for i in range(num_nets):
    Y_pred = Y_pred + models[i].predict(X_val)
    
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val, axis=1) 

# Compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

# Plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes=range(10)) 


# **Plot top losses**

# In[ ]:


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
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    plt.subplots_adjust(hspace=2)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors, axis=1)

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


# # 5. Prediction and submission
# ### Ensembling our results

# In[ ]:


results = np.zeros((X_test.shape[0], 10)) 

for i in range(num_nets):
    results = results + models[i].predict(X_test)
    
results = np.argmax(results, axis=1)
results = pd.Series(results, name='Label')


# ### Submission

# In[ ]:


submission = pd.concat([pd.Series(range(1, X_test.shape[0]+1), name='ImageId'), results], axis=1)
submission.to_csv('ensemble_submission.csv', index=False)

