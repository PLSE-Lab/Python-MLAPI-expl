#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import keras


# # Keras for Neural Networks - Guided Example
# 
# Here we're going to work through a classic Machine Learning problem - digit recognition. This data is referred to as the MNIST dataset, which stands for Modified National Institute of Standards and Technology, and represents probably the most used dataset in the world for advanced machine learning techniques (though the iris dataset would be a close second). Here we're forgoing a more business focussed dataset for a few reasons. Firstly, this dataset is the most written about dataset in these topics - you'll easily find other guides using pure TensorFlow or other tools like Theano to solve the same problem with the same class of models. Similarly, you can also easily find several different kinds of neural networks being used to solve this problem. This will be valuable as you try to expand your knowledge of different kinds of layers and combinations.
# 
# We'll be building our code off of the examples provided in the Keras documentation, and found in full on its [creator's github](https://github.com/fchollet/keras/tree/master/examples). 
# 
# Our goal here will be simple but multifaceted. Overall we are going to use the MNIST dataset and neural networks to classify handwritten numbers as the proper digits. This will be thought of as a multi-class classification problem, specifically with 10 classes (one for each possible digit).
# 
# However, we will use this to teach a few new kinds of neural network compositions, creating three different styles of network and discussing their relative advantages and disadvantages. Through this we will delve a little deeper into neural network theory.
# 
# But before we go too far, let's actually look at the data.
# 
# ## MNIST DATA

# In[ ]:


# Import the dataset
from keras.datasets import mnist

# Import various componenets for model building
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import LSTM, Input, TimeDistributed
from keras.models import Model
from keras.optimizers import RMSprop

# Import the backend
from keras import backend as K

# For Kaggle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# In[ ]:


# # Load Kaggle Data
PATH = '../input/'
test = pd.read_csv(PATH+"test.csv")              #  Sets testing object as a DataFrame
dataset = pd.read_csv(PATH+"train.csv")             #  Set training object as a DataFrame
target = dataset['label'].values.ravel()                #  Set target as the label values flattened to an ndarray(N-dimensional Array)                                            
train = dataset.iloc[:,1:].values                   #  Set train as the pixel values

# # convert to array, specify data type, and reshape
targets = target.astype(np.uint8)
trains = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)
tests = np.array(test).reshape((-1, 1, 28, 28)).astype(np.uint8)

plt.imshow(trains[1729][0], cmap=cm.binary) # draw the picture

plt.show()

y = target                 # Set y equal to label values, ravel strips extra dimensional array 
X = train                    # Set X equal to pixels

print("splitting...")                          # distribute 4 sets
X_train, X_test, Y_train, Y_test = train_test_split(X, 
                    y, test_size=0.5) 


# In[ ]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# The function below plots the images with their labels above
# Code sampled from ../03.%20Dimension%20Reduction.ipynb

import matplotlib.pyplot as plt    #import pylab library which is suited for plotting images

# Set X and y to plot
names = np.sort(dataset.label.unique())     # Created sorted labels array to match titles with images
done = set()                                # Create and empty set of explored indices

#  Create function to plot single digit of interest
def plot_now(images, h=28, w=28, cmap=plt.cm.binary, indx=True, r=0):    # Set default constructor values  
    plt.imshow(images.reshape((h, w)), cmap=cmap)                        # Reshape images and set cmap color
    if indx==True:
        plt.xlabel('index: '+str(r), size=12)                  # Option to show index
        plt.title(names[y][r], size=16)                                      # Sets title from sorted names [y] and index [r] 
    plt.xticks(())                              # Eliminate tick marks
    plt.yticks(())

#  Create function to plot a gallery of interest
def plot_gallery(images, titles,  h=28, w=28, n_row=3, x=1.7, y=2.3, n_col=6, 
                 cmap=plt.cm.binary, random=True, indx=True, r=0, size=16):  
    # Optional row and size parameters will allow us to reuse code, set random to false and r to an index of your choice
    # to see a continous gallery

    plt.figure(figsize=(x * n_col, y * n_row))                          # Set figure size as a ratio of rows
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35) # Adjust spacing of subplots
    for i in range(n_row * n_col):                                         # Adjust spacing of subplots
        move=True                  # logical to move on if a random int is found not in the done set()                                       
        while random and move:      
            r=(np.random.randint(len(images)-(n_row*n_col)-1))          # Create a random integer no greater than the size of the set, and gallery
            if r not in done:                                        # if integer has not been used 
                done.add(r+i)                                       # add to set
            move=False          # Optional randomization to explore sets
        plt.subplot(n_row, n_col, i + 1)                        # create subplot for each
        plt.imshow(images[i+r].reshape((h, w)), cmap=cmap)      # plot images reshaped to 28*28
        if indx==True:plt.xlabel('index: '+str(i+r), size=12)   # print index if True
        plt.title(titles[i+r], size=size)                       # print label
        plt.xticks(())
        plt.yticks(())   # remove ticks


# In[ ]:


plot_gallery(X, names[y], 28, 28, indx=True, random=True, cmap=plt.cm.gray)


# ## Pick out an image that looks weird and plot it below

# In[ ]:


N=17972
plot_now(X[17972], cmap=plt.cm.inferno, indx=False)


# ## Using a mask we can reduce the dimensionality of the data

# In[ ]:


# mask the black pixels
black = np.ma.masked_where(X <= 200, X)

# black = black.compressed()
# blacks = black.reshape(42000, 784).astype(np.uint8).ravel()


# In[ ]:


plot_now(X[N], cmap=plt.cm.gray, indx=False, r=N)


# In[ ]:


weird = set()  # empty set of strange images
N=17972       #
plot_now(black[N], cmap=plt.cm.binary, indx=False, r=N)


# ## Compared with the original, this digit has become more recognizeable through the masking, at least in making its most distinguishing feature more pronounced.

# In[ ]:


plot_gallery(black, names[y],  h=28, w=28, n_row=3, x=1.7, y=2.3, n_col=6, 
                 cmap=plt.cm.binary, random=True, indx=True, r=0, size=16)


# ## Above are random images recreated from the mask at levels of black above 230, as you can see the images have become very simplified, in some cases, over-lossy, but could help combat overfitting.

# In[ ]:


# Let's try reducing dimensionality with PCA to 50 components 
# as we have seen from the eigenvectors from lab 1, 50 should be sufficient
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA.fit_transform
# sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False, 
# svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
from sklearn.decomposition import PCA

print("Fitting PCA...")
n_comp = 50

# whitening was recommended, as well as arpack solver

pca = PCA(n_components=n_comp, whiten=True, svd_solver='arpack')  # Create PCA object

# Set fitted PCA object
trainer = pca.fit_transform(X_train)  # fit and transform the pca model in one operation
tester = pca.fit_transform(X_test)

print("Done!")


# In[ ]:


def pca_plot(X, n_comp, svd_solver='auto', cmap=plt.cm.viridis, scaler=1.0):   # create a plotter function for PCA

    plot_gallery(eigendigits, eigendigit_titles, n_row=int(np.floor(np.sqrt(n_comp))),   # set gallery size to number of PCA to scaler
                 n_col=int(np.ceil(np.sqrt(n_comp))), x=(1.7*scaler), y=(2.3*scaler), 
                 indx=False, random=False, cmap=cmap, size=(16*scaler))


# In[ ]:


try:
    eigendigits = pca.components_.reshape((n_comp, 28, 28))    # set eigenvalues 
    eigendigit_titles = ["eigendigit %d" % i for i in range(eigendigits.shape[0])]  #create the labels
    pca_plot(train, 20, scaler=0.5)
except IndexError:
    pass


# ## The plot above shows the first 20 eigendigits
# The first 10 eigendigits resemble the following digits: 0, 3, 8, 9 After digit 10 they begin to lose form. These will be the most important to train accurately.

# In[ ]:


evr = pca.explained_variance_ratio_         # call evr on the PCA object to get the variance explained by each PC
print( round(sum(evr)*100), "Percent Variance Explained by", n_comp, 'PCs')

# Create cumulative series to plot
cum = 0
d = []
evr = pca.explained_variance_ratio_

for i in range(n_comp):  
    cum += evr[i]
    d.append(cum)


# In[ ]:


sns.set(style='darkgrid')
plt.plot(d, color='orange', label=True)
plt.title('Cumulative Distribution')
plt.xlabel('PC')
plt.ylabel('percent var explained')
plt.show()


# In[ ]:


# n_comp = 100
# # whitening was recommended, as well as arpack solver
# pca = PCA(n_components=n_comp, whiten=True, svd_solver='arpack')  # Create PCA object

# print("fitting...")
# # Set fitted PCA object
# x_train = pca.fit_transform(x_train)  # fit and transform the pca model in one operation
# x_test = pca.fit_transform(x_test)
# print('done!')


# In[ ]:


# Create confusion matrix function to plot errors in predictions
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
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ## try random forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn import metrics as mt
from datetime import datetime as dt
import itertools

start=dt.now()
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)  # 100 estimators has seen the most success
print("Fitting...")
rf.fit(X_train, Y_train)

y_hat = rf.predict(X_test)
conf = mt.confusion_matrix(Y_test,y_hat)
print( 'Accuracy', mt.accuracy_score(Y_test,y_hat)*100, '%')
plot_confusion_matrix(conf, [x for x in range(10)],
                  normalize=False,
                  title='Confusion matrix',
                  cmap=plt.cm.Greens)


# When you look at this data you'll notice its organization structure is not images. We don't actually see any pictures of digits here. Instead, what we have is values of pixels, a simple way of converting images into numeric data on which we can train a model.
# 
# However, this still doesn't look like most of the data we've worked with previously. It's not a single table, but rather a different, higher dimensionality structure. It is often described as a set of clouds, each cloud representing an image. The cloud contains columns of values, representing the darkness of pixels. That's great, but not an easy or meaningful dataset on which to directly train a model. The darkness of the second pixel in the third column isn't likely linearly related to likelihood the cloud represents a certain digit. Instead, we need to find meaningful patterns within our clouds, creating models off of those patterns.
# 
# This is exactly what neural networks are good at. Multiple layers will allow us to transform this clouds full of values into meaningful vectors containing the information we need to be able to create a model, admittedly in an unlabeled and unsupervised fashion. Our output, however, will be labels for each of the clouds, giving us predictions as to what digit they are meant to represent.
# 
# Let's get started.

# ## Multi Layer Perceptron
# 
# Let's start with a kind of neural network we've seen before: a multi-layer perceptron. Recall from our previous neural networks sections that this is a set of perceptron models organized into layers, one layer feeding into the next.
# 
# To do this, we will first need to reshape our data into flat vectors for each digit. We'll also need to convert our outcome to a matrix of binary variables, rather than the digit.

# In[ ]:


# Change shape 
# Note that our images are 28*28 pixels, so in reshaping to arrays we want
# 60,000 arrays of length 784, one for each image


x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

# Convert to float32 for type consistency
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize values to 1 from 0 to 255 (256 values of pixels)
x_train /= 255
x_test /= 255

# Print sample sizes
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
# So instead of one column with 10 values, create 10 binary columns
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# Great. Now we can create our model. We'll do this using dense layers and dropouts. Dense layers are simply fully connected layers with a given number of perceptrons. Dropout drops a certain portion of our perceptrons in order to prevent overfitting. Our activation function, `relu` stands for Rectified Linear Unit, which is standard but can be read about more [here](https://en.wikipedia.org/wiki/Rectifier_(neural_networks).

# In[ ]:


# # Start with a simple sequential model
# model = Sequential()

# # Add dense layers to create a fully connected MLP
# # Note that we specify an input shape for the first layer, but only the first layer.
# # Relu is the activation function used
# model.add(Dense(64, activation='relu', input_shape=(784,)))
# # Dropout layers remove features and fight overfitting
# model.add(Dropout(0.1))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.1))
# # End with a number of units equal to the number of classes we have for our outcome
# model.add(Dense(10, activation='softmax'))

# model.summary()

# # Compile the model to put it all together.
# model.compile(loss='categorical_crossentropy',
#               optimizer=RMSprop(),
#               metrics=['accuracy'])


# Now we have a model. This we can use to accomplish our wildest dreams of data modeling, or at least predict some digits from pixel data. To do that we will use epochs, effectively iterations of the model, improving based on what it learned previously. Batch size is the number of samples to use in each step improving the model and will affect speed, but also slightly negatively impact accuracy (learning in bigger steps will affect what your model learns).
# 
# Note that we are going with 64 perceptron wide layers, this is relatively arbitrary, though units within the $2^x$ series will parallelize more efficiently. Also note that our number of parameters is the product of our input width plus one and our layer width. This reflects the number of weights we're creating in that layer.

# In[ ]:


# history = model.fit(x_train, y_train,
#                     batch_size=128,
#                     epochs=10,
#                     verbose=1,
#                     validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])


# That did impressively well for such a simple neural network, with each epoch training in about 1 second on this machine and giving us an accuracy in the high 90's. But what else can we do? Let's let our model get much more complicated by introducting convolution.
# 
# ## Convolutional Neural Networks
# 
# Before we go any further, do you recall that we've talked about how complex neural networks can get, and the degree of computational complexity that entails? Well, here we're going to finally truly experience that complexity, so be careful about rerunning this code. It will take some serious time (potentially on the order of hours) to run.
# 
# Now that that's out of the way, let's talk convolution. First, a simple definition. Convolution basically takes your data and creates overlapping subsegments testing for a given feature in a set of spaces and upon which it develops its model.
# 
# Let's extend that definition since it's incredibly dense.
# 
# First, you have to define a shape of your input data. This can theoretically be in any number of dimensions, though for our image example we will use 2d, since images are in two dimensions. This is also why you'll see 2D in some of our layer definitions (though more on that later). Our first chunk of code after loading the data does this reshaping (with a conditional on the data format).
# 
# Over that shaped data, we then create our tiles, also called __kernels__. These kernels are like little windows, that will look over subsets of the data of a given size. In the example below we create 3x3 kernels, which run overlapping over the whole 28x28 input looking for features. That is the convolutional layer, a way of searching for a subpattern over the whole of the image. We can chain multiple of these convolutional layers together, with the below example having two.
# 
# Next comes a pooling layer. This is a _downsampling_ technique, which effectively serves to reduce sample size and simplify later processes. For each value generated by our convolutional layers, it looks over the grid in _non_-overlapping segments and takes the maximum value of those outputs. It's not the feautres exact location then that matters, but its approximate or relative location. After pooling you will want to flatten the data back out, so that it can be put into dense layers as we did in MLP.

# In[ ]:


# # input image dimensions, from our data
# img_rows, img_cols = 28, 28
# num_classes = 10

# # the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)


# # Building the Model
# model = Sequential()
# # First convolutional layer, note the specification of shape
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])

# model.fit(x_train, y_train,
#           batch_size=128,
#           epochs=10,
#           verbose=1,
#           validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])


# Now that is incredibly impressive accuracy. 99% is really exceptional, but it did take a long time to get there. Such are the costs of convolution.
# 
# There is one more classic construction of a neural network: Recurrent, which we'll give quick mention.
# 
# ## Hierarchical Recurrrent Neural Networks
# 
# So far when we've talked about neural networks we've talked about them as feedforward: data flows in one direction until it reaches the end. Recurrent neural networks do not obey that directional logic, instead letting the data cycle through the network.
# 
# However, to do this we have to abandon the sequential model building we've done so far and things can get much more complicated. You have to use recurrent layers and often time distribution (which handles the extra dimension created through the LTSM layer, as a time dimension) to get these things running. You can find an example of a hierarchical recurrent network below (via the link [here](https://github.com/fchollet/keras/blob/master/examples/mnist_hierarchical_rnn.py)). When you get comfortable with networks as they exist in Keras for both convolution and MLP, start exploring recurrence. Note that this will take an even longer time than the previous ones should you choose to run it again.

# In[ ]:



# # Training parameters.
# batch_size = 64
# num_classes = 10
# epochs = 3

# # Embedding dimensions.
# row_hidden = 32
# col_hidden = 32

# # The data, shuffled and split between train and test sets.
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Reshapes data to 4D for Hierarchical RNN.
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# # Converts class vectors to binary class matrices.
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

# row, col, pixel = x_train.shape[1:]

# # 4D input.
# x = Input(shape=(row, col, pixel))

# # Encodes a row of pixels using TimeDistributed Wrapper.
# encoded_rows = TimeDistributed(LSTM(row_hidden))(x)

# # Encodes columns of encoded rows.
# encoded_columns = LSTM(col_hidden)(encoded_rows)

# # Final predictions and model.
# prediction = Dense(num_classes, activation='softmax')(encoded_columns)
# model = Model(x, prediction)
# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

# # Training.
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))

# # Evaluation.
# scores = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])


# You should now be comfortable building some neural networks, but let's see if you can improve them!
# 
# # Drill: 99% MLP
# 
# We have the MLP above, which runs reasonably quickly. Copy that code down here and see if you can tune it to achieve 99% accuracy with a Multi-Layer Perceptron. Does it run faster than the recurrent or concolutional neural nets?

# In[ ]:


# Start with a simple sequential model
model = Sequential()

# Add dense layers to create a fully connected MLP
# Note that we specify an input shape for the first layer, but only the first layer.
# Relu is the activation function used
model.add(Dense(256, activation='relu', input_shape=(784,)))
# Dropout layers remove features and fight overfitting
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
# End with a number of units equal to the number of classes we have for our outcome
model.add(Dense(10, activation='softmax'))

model.summary()

# Compile the model to put it all together.
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


# In[ ]:


model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


# import test set from kaggle and test from trained MNIST data
X_test.shape


# In[ ]:


X_test = test.values.reshape(28000, 28, 28, 1)


# In[ ]:


preds = model.predict_classes(test, verbose=0)


# In[ ]:


# my_submission = pd.DataFrame(preds).copy()
# my_submission.to_csv('submission_KerasMLP2.csv', index=False)


# ## save predictions for submission

# In[ ]:


np.savetxt('submission_KerasMLP2.csv', np.c_[range(1,len(test)+1),preds], 
           delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')


# In[ ]:




