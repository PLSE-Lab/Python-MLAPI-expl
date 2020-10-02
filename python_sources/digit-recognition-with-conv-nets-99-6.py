#!/usr/bin/env python
# coding: utf-8

# # Welcome!
# In this kernel I will walk you through some basic exploratory data analysis (EDA) and how to set up a simple convolutional neural network to perform image classification. Hope you find this helpful. Please feel free to comment below if you have any questions.

# ## 1. Setting up our environment
# 
# Let's get started by setting up our environment. We will be using the Keras environment to set up our convolutional neural network (CNN). In addition to Keras we will be using numpy and pandas to help us manipulate and set up our data for our model. Finally, we will be using matplotlib and seaborn for plotting and visualization If you are unfamiliar with any of these, please take advantage of the links below.
# 
# * References:
#     * Keras: https://keras.io/
#     * Numpy: https://docs.scipy.org/doc/
#     * Pandas: http://pandas.pydata.org/pandas-docs/stable/
#     * Matplotlib: https://matplotlib.org/index.html
#     * Seaborn: https://seaborn.pydata.org/

# In[ ]:


# Import numpy and pandas
import numpy as np 
import pandas as pd

# We'll use this later for splitting our data up into training and validation sets
from sklearn.model_selection import train_test_split

# Import matplotlib and sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Get everything we need from Keras
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Dropout
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
K.set_image_data_format('channels_last')


# ## 2. Loading our data and exploring it
# So now that we have our environment set up, let's load our data and start exploring it. First we use pandas to read in the data. The data provided here is stored in the form of a CSV file. Each row consists of a label (0, 1, 2, 3, 5, ..., 9) and 748 pixel intensity values. Run the code in the cell below this to load the data.

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# Our data is now stored as a pandas 'DataFrame'. We can take advantage of the 'head' attribute to get a glimpse at what our raw data looks like.

# In[ ]:


train.head(5)


# Next on the adgenda is to see how many examples of each class we have. For this we use the 'countplot' function available through Seaborn. Go ahead and run the code below. We can see that we have roughly the same amount of training examples for each class. This is good! When training a model you want to have a uniform amount of examples from each class. If this is not the case, then you have a class imbalance problem. There are several methods available for fixing that issue. See the link below for more on that.
# 
# * References:
#     * Countplot: https://seaborn.pydata.org/generated/seaborn.countplot.html?highlight=countplot#seaborn.countplot
#     * Class imbalance problem: https://towardsdatascience.com/dealing-with-imbalanced-classes-in-machine-learning-d43d6fa19d2
# 
# 

# In[ ]:


# Get number of training examples
print('Number of training examples: ' + str(train.shape[0]))

# Plot the number of examples per class in our training set
sns.countplot(train['label'])


# We have our data loaded and we know that it is evenly distributed among our different classes, but what does each example actually look like? Let's find out! The code below generates a set of 25 random examples, reshapes those examples from a 1 x 748 array to a 28 x 28 array, and then displays that array using matplotlib's 'imshow' function. 
# 
# * References:
#     * np.reshape: https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
#     * imshow: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html

# In[ ]:


# Get a list of 25 random examples
example_ids = np.random.randint(28000, size = 25)

# Set up your figure
cnt = 1
fig = plt.figure()
title = fig.suptitle('MNIST Handwritten Digit Examples', fontsize = 20)
title.set_position([.5, .92])
fig.set_figheight(12)
fig.set_figwidth(10)
for id in example_ids:
    # Get the data associated with each example
    example_data = train.iloc[[id]]
    
    # Get the label associated with each example
    example_truth = example_data['label'].unique()[0]
    
    # Drop the label so we are only left with the image data
    example_data = example_data.drop(['label'], axis = 1)
    
    # Convert the data to a numpy array so that it's compatible with imshow
    example_data = np.array(example_data)
    
    # Reshape the example to a 28 x 28 array
    example_data = np.reshape(example_data, [28, 28])
    
    # Plot that array in a subplot
    plt.subplot(5, 5, cnt)
    plt.title('Label = ' + str(np.array(example_truth)))
    plt.axis('off')
    plt.imshow(example_data)
    cnt += 1


# ## 3. Setting up our CNN
# Now for the fun part! Let's go ahead and set up the CNN that we will be using to perform image classification. Our CNN will have the following structure:
# 
# INPUT - > CONV -> BN -> RELU -> CONV -> BN -> RELU -> MAXPOOL ->  DROPOUT -> CONV -> BN -> RELU -> CONV -> BN -> RELU -> MAXPOOL -> DROPOUT -> FULLY CONNECTED -> DROPOUT -> FULLY CONNECTED -> DROPOUT -> SOFTMAX
# 
# If you are unfamiliar with concepts such as convolutional layers, batch normalization, relu, and pooling, I would recommend taking the following course on CNNs.
# 
# * References:
#     * Convolutional Neural Networks: https://www.coursera.org/learn/convolutional-neural-networks

# In[ ]:


def digit_conv_net(input_shape, num_classes):
    
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((2, 2))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (5, 5), strides = (1, 1), name = 'conv0', padding = 'same', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (5, 5), strides = (1, 1), name = 'conv1', padding = 'same', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D(pool_size = (2, 2), name = 'max_pool0')(X)
    
    # Add dropout to regularize
    X = Dropout(0.25)(X)
    
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(64, (5, 5), strides = (1, 1), name = 'conv2', padding = 'same', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(64, (5, 5), strides = (1, 1), name = 'conv3', padding = 'same', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D(pool_size = (2, 2), name = 'max_pool1')(X)
    
    # Add dropout to regularize
    X = Dropout(0.25)(X)
    
    # FLATTEN X -> FULLY CONNECTED
    X = Flatten()(X)
    X = Dense(128, activation = 'relu', name = 'fc0')(X)
    X = Dropout(0.25)(X)
    X = Dense(128, activation = 'relu', name = 'fc1')(X)
    X = Dropout(0.25)(X)
    X = Dense(num_classes, activation = 'softmax', name = 'fc2')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name = 'LeNet5')

    return model


# We've defined the basic structure of our CNN. Now let's initilize it with the input size and number of classes. Once that is done, we complie the model.
# 
# * References
#     * Compile: https://keras.io/models/model/

# In[ ]:


# Initialize model with input size and number of classes
DigitModel = digit_conv_net([28, 28, 1], 10)

# Complie the model. Use Adam optimizer.
DigitModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# You can use the 'summary' attribute to see the structure of your model and get the number of parameters you will be training.
# 
# * References:
#     * Summary: https://keras.io/models/about-keras-models/#about-keras-models

# In[ ]:


DigitModel.summary()


# ### 3.1. Preprocessing our data
# 
# Our model has now been initialized and compiled. Now we need to set up our data so that we can feed it into the model. Do do this we separate our labels from our pixel intensity values. Our labels will become our target (stored as y) and our intensity values will be reshaped into 28 x 28 images. Before that happens, however, we will use the 'train_test_split' function to break our data into a training set and a validation set. Finally, we divide our pixel intensities by 255 to normalize them. This last step scales the data so that every pixel value is between 0 and 1. This speeds up optimization.
# 
# * References:
#     * train_test_split: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

# In[ ]:


train_labels = train['label']
train_labels = pd.get_dummies(train_labels)
train_features = train.drop(['label'], axis = 1)
X_train, X_val, y_train, y_val = train_test_split(train_features,
                                                   train_labels,
                                                   train_size = 0.98,
                                                   test_size = 0.02,
                                                   random_state = 0)
X_train = np.array(X_train)
X_train = np.reshape(X_train, [X_train.shape[0], 28, 28, 1]) / 255.

X_val = np.array(X_val)
X_val = np.reshape(X_val, [X_val.shape[0], 28, 28, 1]) / 255.

test = np.array(test)
test = np.reshape(test, [test.shape[0], 28, 28, 1]) / 255.


# ### 3.2. Some finishing touches...callbacks and data augmentation
# 
# At this point, our data and our model are ready to go. However, before we go any further, I'd like to showcase two particularly nice features in Keras: Callbacks and real-time data augmentation. A callback is a set of functions to be applied at given stages of the training procedure. You can use callbacks to get a view on internal states and statistics of the model during training. Here we use the 'ReduceLROnPlateau callback. In a nutshell, this monitors a parameter of our choice (i.e. the validation accuracy) and reduces the learning rate for our optimizer if our parameter starts to plateau. In this instance, the following inputs are given:
# 
# * monitor - What parameter our model will monitor
# * factor - What we multiply the learning rate by in the event of a plateau
# * patience - How many epochs we are willing to wait before we reduce the learning rate
# * min_lr - How low we are willing to let the learning rate go. Do not set this too low. It will slow down your optimizer.
# 
# * Refernces:
#     * Callbacks: https://keras.io/callbacks/

# In[ ]:


ReduceLearningRate = ReduceLROnPlateau(monitor = 'val_acc', 
                                       factor = 0.5, 
                                       patience = 3, 
                                       min_lr = 0.00001)


# Data augmentation is the process of applying small changes to our existing data set to generate more data. Take the kitten example below. From the original image, we can change the height and width, rotate, crop, and flip it to generate more training examples. Keras has a function called 'ImageDataGenerator' that applies these transformations for us in real time. These new examples are fed into our model during training. I would highly recommend going to the link below to see what options are available on Keras.
# 
# * References:
#     * ImageDataGenerator: https://keras.io/preprocessing/image/
# 
# ![](https://cdn-images-1.medium.com/max/1600/1*C8hNiOqur4OJyEZmC7OnzQ.png)
# 
# Image source: https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced

# In[ ]:


DataGenerator = ImageDataGenerator(rotation_range = 10,
                                   zoom_range = 0.1,
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1)

# Fits the model on batches with real-time data augmentation
DataGenerator.fit(X_train)


# ## 4. Fire it all up!
# 
# Now we are ready to go. We've defined our model, precprocessed our data, defined a callback to monitor the progress of our model, and we've set our model up with data augmentation. Run the block of code below and enjoy the show. For better performance, increase the number of epochs (25-30 epochs should do the trick).

# In[ ]:


# Store this in a variable called mdl which we will use later to visualize the model performance
mdl = DigitModel.fit_generator(DataGenerator.flow(X_train, y_train, batch_size = 32), 
                               epochs = 15,
                               validation_data = (X_val, y_val),
                               callbacks = [ReduceLearningRate])


# ## 5. Checking our work
# 
# Our model is fully trained and we are ready to run our test data through it. However, before we do that, we can use the block of code below to check for any underfitting or overfitting. Earlier we split the data up into a training and validation sets. Below you have two plots. One shows training and validation accuracy vs epochs, and the other shows training and validation loss vs epochs. Ideally you want to see your training and validation accuracy go up AND remain close to each other. Similarly, you want to see your training and validation loss go down AND remain close to each other. This lets you know that your model is generalizing well to data that it's never seen before.
# 

# In[ ]:


fig = plt.figure()
title = fig.suptitle('Model Accuracy/Loss Summary', fontsize = 16)
title.set_position([.5, 1.0])
fig.set_figheight(5)
fig.set_figwidth(10)

# Summarize history for accuracy
plt.subplot(1, 2, 1)
plt.plot(mdl.history['acc'])
plt.plot(mdl.history['val_acc'])
plt.title('Model Accuracy', fontsize = 10)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

# Summarize history for loss
plt.subplot(1, 2, 2)
plt.plot(mdl.history['loss'])
plt.plot(mdl.history['val_loss'])
plt.title('Model Loss', fontsize = 10)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')


# ## 6. Submit your work!
# 
# You can see above that our model is doing very well! Both the loss and and accuracy for our training and validation sets are similar. Feel free to use this code or play around with it even more. Hope this helps, and please feel free to leave any questions in the comments section.

# In[ ]:


# predict results
results = DigitModel.predict(test)

# select the index with the maximum probability
results = np.argmax(results, axis = 1)
results = pd.Series(results, name = "Label")


# In[ ]:


my_submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"), results], axis = 1)
my_submission.to_csv("my_submission.csv", index = False)


# In[ ]:




