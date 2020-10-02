#!/usr/bin/env python
# coding: utf-8

# # A Beginner's Guide - CNN with Keras
# 
# Kaiming Kuang
# 
# This is a beginner's guide of the Digit Recognizer competition. Some basic knowledges about the theory and practice of deep learning is still required. Here are some prerequisite readings on Convolutional Neural Network:
# - [Wikipedia of CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)
# - [A Beginner's Guide To Understanding Convolutional Neural Networks by Adit Deshpande](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)
# 
# There is also a course on Coursera. I learned most of the basics from this course of Andrew Ng:
# - [Coursera: Convolutional Neural Network by Andrew Ng](https://www.coursera.org/learn/convolutional-neural-networks?specialization=deep-learning)
# 
# ## Content
# - 1 Introduction
# - 2 Data Exploration, Augmentation and Preparation
#     - 2.1 Data Exploration
#     - 2.2 Data Augmentation
#         - 2.2.1 Zoom In
#         - 2.2.2 Translation
#         - 2.2.3 Add White Noise
#         - 2.2.4 Rotation
#     - 2.3 Data Preparation
# - 3 CNN Structure
# - 4 Training and Evaluation
#     - 4.1 Train the Model
#     - 4.2 Evaluate the Model
# - 5 Output the Prediction

# # 1 Introduction
# 
# The Digit Recognizer competition uses the famous MNIST hand-written number dataset, which is the hello-world dataset for computer vision. Here we are required to identify numbers from images. The training data contains 42,000 hand-written number images, each one of which is 28 pixels in height and 28 pixels in width. Simple models such as KNN or MLP are not as capable of this task as Convolutional Neural Network. Here I used Keras to build my CNN, for that it is more friendly to beginners than TensorFlow and these two are the only two DL frameworks that I am familiar with. Please do turn on the GPU button when running this kernel because it is extremely time-consuming to run CNN on CPU. It usually took me 50+ hours to run this code for 100 epochs on my laptop. With the Tesla K80 GPU, 100 epochs took only a little more than 1 hour. With early stopping, sometimes it only took around 15 mins to run

# First we should import all the libraries we need in this kernel:
# - gc: The built-in garbage collection of Python. We need to delete some variables and collect spaces when necessary to save RAM.
# - random: The built-in package of Python. We need it to generate random numbers.
# - time: The built-in package of Python. Use it to check running time.
# - pi: In the data augmentation part we use pi to rotate the image.
# - keras: We need Keras to build our CNN model. It uses TensorFlow as backend. [Documentation of Keras](https://keras.io/).
# - matplotlib.pyplot: We use pyplot to plot the hand-written number image.
# - numpy: We need Numpy to do all the matrix manipulation. [Documentation of Numpy](https://docs.scipy.org/doc/numpy/reference/).
# - pandas: We use Pandas to manipulate data, such as loading and outputing .csv files. [Documentation of Pandas](http://pandas.pydata.org/pandas-docs/stable/).
# - tensorflow: TensorFlow is a popular deep learning framework. We use TensorFlow for the data augmentation part. [Documentation of TensorFlow](https://tensorflow.google.cn/api_docs/python/tf).
# - ReduceLROnPlateau: This is the model we use to set up a learning rate decay.
# - BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPool2D: These are some basic building blocks we need to set up CNN.
# - Image: This package is used for image display.
# - train_test_split: We use this module of sklearn to split the data into trainning and validation part.

# In[ ]:


import gc
import random as rd
import time
from math import pi

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          MaxPool2D, ReLU)
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# # 2 Data Exploration, Augmentation and Preparation

# ## 2.1 Data Exploration
# 
# First load the data.

# In[ ]:


print("Loading...")
data_train = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")
print("Done!")


# In[ ]:


print("Training data: {} rows, {} columns.".format(data_train.shape[0], data_train.shape[1]))
print("Test data: {} rows, {} columns.".format(data_test.shape[0], data_test.shape[1]))


# There are 42,000 rows in the training data and 28,000 rows in the test data. Each row of the training set contains the image (28x28=784) and the label in the first column. The test data doesn't have the labels.

# In[ ]:


x_train = data_train.values[:, 1:]
y_train = data_train.values[:, 0]


# Now let's see how the numbers look like. We will use the convert_2d function to convert the 1d data into two dimensions.

# In[ ]:


def convert_2d(x):
    """x: 2d numpy array. m*n data image.
       return a 3d image data. m * height * width * channel."""
    if len(x.shape) == 1:
        m = 1
        height = width = int(np.sqrt(x.shape[0]))
    else:
        m = x.shape[0]
        height = width = int(np.sqrt(x.shape[1]))

    x_2d = np.reshape(x, (m, height, width, 1))
    
    return x_2d


# In[ ]:


x_display = convert_2d(data_train.values[0, 1:])
plt.imshow(x_display.squeeze(), cmap="gray")


# ## 2.2 Data Augmentation
# 
# Here we delve straight into data augmentation. Data augmentation is a useful technique when you don't have enough data or would like to expand your data to improve the performance. In this competition, data augmentation basically means cutting, rotating and zooming the image without hurting its identifiability. Here I used zooming, translation, white noise and rotation. With data augmentation, you can expect a 1-2% accuracy improvement.

# ### 2.2.1 Zoom In
# Here we use crop_image function to crop a part of the image around the center, resize it and save it as augmented data.

# In[ ]:


def crop_image(x, y, min_scale):
    """x: 2d(m*n) numpy array. 1-dimension image data;
       y: 1d numpy array. The ground truth label;
       min_scale: float. The minimum scale for cropping.
       return zoomed images.
       This function crops the image, enlarges the cropped part and uses it as augmented data."""
    # convert the data to 2-d image. images should be a m*h*w*c numpy array.
    images = convert_2d(x)
    # m is the number of images. Since this is a gray-scale image scale from 0 to 255, it only has one channel.
    m, height, width, channel = images.shape
    
    # tf tensor for original images
    img_tensor = tf.placeholder(tf.int32, [1, height, width, channel])
    # tf tensor for 4 coordinates for corners of the cropped image
    box_tensor = tf.placeholder(tf.float32, [1, 4])
    box_idx = [0]
    crop_size = np.array([height, width])
    # crop and resize the image tensor
    cropped_img_tensor = tf.image.crop_and_resize(img_tensor, box_tensor, box_idx, crop_size)
    # numpy array for the cropped image
    cropped_img = np.zeros((m, height, width, 1))

    with tf.Session() as sess:

        for i in range(m):
            
            # randomly select a scale between [min_scale, min(min_scale + 0.05, 1)]
            rand_scale = np.random.randint(min_scale * 100, np.minimum(min_scale * 100 + 5, 100)) / 100
            # calculate the 4 coordinates
            x1 = y1 = 0.5 - 0.5 * rand_scale
            x2 = y2 = 0.5 + 0.5 * rand_scale
            # lay down the cropping area
            box = np.reshape(np.array([y1, x1, y2, x2]), (1, 4))
            # save the cropped image
            cropped_img[i:i + 1, :, :, :] = sess.run(cropped_img_tensor, feed_dict={img_tensor: images[i:i + 1], box_tensor: box})
    
    # flat the 2d image
    cropped_img = np.reshape(cropped_img, (m, -1))
    cropped_img = np.concatenate((y.reshape((-1, 1)), cropped_img), axis=1).astype(int)

    return cropped_img


# ### 2.2.2 Translation
# Now we shift the image to 4 different directions. 

# In[ ]:


def translate(x, y, dist):
    """x: 2d(m*n) numpy array. 1-dimension image data;
       y: 1d numpy array. The ground truth label;
       dist: float. Percentage of height/width to shift.
       return translated images.
       This function shift the image to 4 different directions.
       Crop a part of the image, shift it and fill the left part with 0."""
    # convert the 1d image data to a m*h*w*c array
    images = convert_2d(x)
    m, height, width, channel = images.shape
    
    # set 4 groups of anchors. The first 4 int in a certain group lay down the area we crop.
    # The last 4 sets the area to be moved to. E.g.,
    # new_img[new_top:new_bottom, new_left:new_right] = img[top:bottom, left:right]
    anchors = []
    anchors.append((0, height, int(dist * width), width, 0, height, 0, width - int(dist * width)))
    anchors.append((0, height, 0, width - int(dist * width), 0, height, int(dist * width), width))
    anchors.append((int(dist * height), height, 0, width, 0, height - int(dist * height), 0, width))
    anchors.append((0, height - int(dist * height), 0, width, int(dist * height), height, 0, width))
    
    # new_images: d*m*h*w*c array. The first dimension is the 4 directions.
    new_images = np.zeros((4, m, height, width, channel))
    for i in range(4):
        # shift the image
        top, bottom, left, right, new_top, new_bottom, new_left, new_right = anchors[i]
        new_images[i, :, new_top:new_bottom, new_left:new_right, :] = images[:, top:bottom, left:right, :]
    
    new_images = np.reshape(new_images, (4 * m, -1))
    y = np.tile(y, (4, 1)).reshape((-1, 1))
    new_images = np.concatenate((y, new_images), axis=1).astype(int)

    return new_images


# ### 2.2.3 Add White Noise
# Now we add some white noise to the image. We randomly choose some pixels and replace them with uniformly-distributed noise.

# In[ ]:


def add_noise(x, y, noise_lvl):
    """x: 2d(m*n) numpy array. 1-dimension image data;
       y: 1d numpy array. The ground truth label;
       noise_lvl: float. Percentage of pixels to add noise in.
       return images with white noise.
       This function randomly picks some pixels and replace them with noise."""
    m, n = x.shape
    # calculate the # of pixels to add noise in
    noise_num = int(noise_lvl * n)

    for i in range(m):
        # generate n random numbers, sort it and choose the first noise_num indices
        # which equals to generate random numbers w/o replacement
        noise_idx = np.random.randint(0, n, n).argsort()[:noise_num]
        # replace the chosen pixels with noise from 0 to 255
        x[i, noise_idx] = np.random.randint(0, 255, noise_num)

    noisy_data = np.concatenate((y.reshape((-1, 1)), x), axis=1).astype("int")

    return noisy_data


# ### 2.2.4 Rotation
# Now we rotate the image.

# In[ ]:


def rotate_image(x, y, max_angle):
    """x: 2d(m*n) numpy array. 1-dimension image data;
       y: 1d numpy array. The ground truth label;
       max_angle: int. The maximum degree for rotation.
       return rotated images.
       This function rotates the image for some random degrees(0.5 to 1 * max_angle degree)."""
    images = convert_2d(x)
    m, height, width, channel = images.shape
    
    img_tensor = tf.placeholder(tf.float32, [m, height, width, channel])
    
    # half of the images are rotated clockwise. The other half counter-clockwise
    # positive angle: [max/2, max]
    # negative angle: [360-max/2, 360-max]
    rand_angle_pos = np.random.randint(max_angle / 2, max_angle, int(m / 2))
    rand_angle_neg = np.random.randint(-max_angle, -max_angle / 2, m - int(m / 2)) + 360
    rand_angle = np.transpose(np.hstack((rand_angle_pos, rand_angle_neg)))
    np.random.shuffle(rand_angle)
    # convert the degree to radian
    rand_angle = rand_angle / 180 * pi
    
    # rotate the images
    rotated_img_tensor = tf.contrib.image.rotate(img_tensor, rand_angle)

    with tf.Session() as sess:
        rotated_imgs = sess.run(rotated_img_tensor, feed_dict={img_tensor: images})
    
    rotated_imgs = np.reshape(rotated_imgs, (m, -1))
    rotated_imgs = np.concatenate((y.reshape((-1, 1)), rotated_imgs), axis=1)
    
    return rotated_imgs


# Now we put them all together.

# In[ ]:


start = time.clock()
print("Augment the data...")
cropped_imgs = crop_image(x_train, y_train, 0.9)
translated_imgs = translate(x_train, y_train, 0.1)
noisy_imgs = add_noise(x_train, y_train, 0.1)
rotated_imgs = rotate_image(x_train, y_train, 10)

data_train = np.vstack((data_train, cropped_imgs, translated_imgs, noisy_imgs, rotated_imgs))
np.random.shuffle(data_train)
print("Done!")
time_used = int(time.clock() - start)
print("Time used: {}s.".format(time_used))


# ## 2.3 Data Preparation

# Let's check the augmented data.

# In[ ]:


x_train = data_train[:, 1:]
y_train = data_train[:, 0]
x_test = data_test.values
print("Augmented training data: {} rows, {} columns.".format(data_train.shape[0], data_train.shape[1]))


# Now we need to convert the 1d image data to 2-dimension.

# In[ ]:


x_train = convert_2d(x_train)
x_test = convert_2d(x_test)


# Also, we need the label variable to be a dummy variable, which only contains 1 and 0. We will use a Keras utility function to do the conversion.

# In[ ]:


num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)


# The values in the image range from 0 to 255. It would be easier for CNN to converge if we scale down these values. Thus, we divide all pixels by 255.

# In[ ]:


x_train = x_train / 255
x_test = x_test / 255


# Now we split the dataset into the training set and the developing(validation) set.

# In[ ]:


# generate a random seed for train-test-split
seed = np.random.randint(1, 100)
x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)


# Delete the original data_train to save some RAM.

# In[ ]:


del data_train
del data_test
gc.collect()


# # 3 CNN Structure
# 
# A normal CNN usually consists of 3 types of layers, convolutional layers, pooling layers and fully-connected layers. I also added normalization layers and dropout layers into my model. Here is how I set up the CNN structure.

# In[ ]:


# number of channels for each of the 4 convolutional layers. 
filters = (32, 32, 64, 64)
# I use a 5x5 kernel for every conv layer
kernel = (5, 5)
# the drop probability of the dropout layer
drop_prob = 0.2

model = keras.models.Sequential()

model.add(Conv2D(filters[0], kernel, padding="same", input_shape=(28, 28, 1),
                 kernel_initializer=keras.initializers.he_normal()))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Conv2D(filters[0], kernel, padding="same",
                 kernel_initializer=keras.initializers.he_normal()))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPool2D())
model.add(Dropout(drop_prob))

model.add(Conv2D(filters[1], kernel, padding="same",
                 kernel_initializer=keras.initializers.he_normal()))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPool2D())
model.add(Dropout(drop_prob))

model.add(Conv2D(filters[2], kernel, padding="same",
                 kernel_initializer=keras.initializers.he_normal()))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPool2D())
model.add(Dropout(drop_prob))

model.add(Conv2D(filters[3], kernel, padding="same",
                 kernel_initializer=keras.initializers.he_normal()))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPool2D())
model.add(Dropout(drop_prob))

# several fully-connected layers after the conv layers
model.add(Flatten())
model.add(Dropout(drop_prob))
model.add(Dense(128, activation="relu"))
model.add(Dropout(drop_prob))
model.add(Dense(num_classes, activation="softmax"))
# use the Adam optimizer to accelerate convergence
model.compile(keras.optimizers.Adam(), "categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


model.summary()


# The list above is the structure of my CNN model. It goes:
# - (Conv-ReLU-BatchNormalization-MaxPooling-Dropout) x 4;
# - 3 fully-connected(dense) layers with 1 dropout layer. Dense(64)-Dense(128)-Dropout-Dense(with softmax activation).
# 
# - In CNN people often use 3x3 or 5x5 kernel. I found that with a 5x5 kernel, the model's accuracy improved about 0.125%, which is quite a lot when you pass 99% threshold.
# - Convolutional layers and max pooling layers can extract some high-level traits from the pixels. With the [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) unit the and max pooling, we also add non-linearity into the network;
# - Batch normalization helps the network converge faster since it keeps the input of every layer at the same scale;
# - [Dropout](https://en.wikipedia.org/wiki/Convolutional_neural_network#Dropout) layers help us prevent overfitting by randomly drop some of the input units. With dropout our model won't overfit to some specific extreme data or some noisy pixels;
# - The [Adam optimizer](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) also accelerates the optimization. Usually when the dataset is too large, we use mini-batch gradient descent or stochastic gradient descent to save some training time. The randomness in MBGD or SGD means that the steps towards the optimum are zig-zag rather than straight forward. Adam, or Adaptive Moment Estimation, uses exponential moving average on the gradients and the secend moment of gradients to make the steps straight and in turn accelerate the optimization.

# # 4 Training and Evaluation

# ## 4.1 Train the Model
# Now we need to train our model. First let's set some basic hyperparameters for training.

# In[ ]:


# number of epochs we run
iters = 100
# batch size. Number of images we train before we take one step in MBGD.
batch_size = 1024


# In Andrew Ng's deep learning course, he mentioned that it would be better to set the batch size to the power of 2 due to some reasons regarding hardware or TensorFlow underlying code. Not so sure about that.

# When we reach close to the optimum, we need to lower our learning rate to prevent overshooting. Large learning rate would keep us away from the optimum. Thus, I set this learning rate decay to decrease it when the accuracy on the validation data no longer improves.

# In[ ]:


# monitor: the quantity to be monitored. When it no longer improves significantly, we lower the learning rate
# factor: new learning rate = old learning rate * factor
# patience: number of epochs we wait before we decrease the learning rate
# verbose: whether or not the message are displayed
# min_lr: the minimum learning rate
lr_decay = ReduceLROnPlateau(monitor="val_acc", factor=0.5, patience=3, verbose=1, min_lr=1e-5)


# If our model are not getting any better on the validation data, we can set early stopping to prevent overfitting and also save some time. Early stopping stops the training when the monitored quantity doesn't improve.

# In[ ]:


# monitor: the quantity to be monitored. When it no longer improves significantly, stop training
# patience: number of epochs we wait before training is stopped
# verbose: whether or not to display the message
early_stopping = EarlyStopping(monitor="val_acc", patience=7, verbose=1)


# Now we train the model.

# In[ ]:


print("Training model...")
fit_params = {
    "batch_size": batch_size,
    "epochs": iters,
    "verbose": 1,
    "callbacks": [lr_decay, early_stopping],
    "validation_data": (x_dev, y_dev)                   # data for monitoring the model accuracy
}
history = model.fit(x_train, y_train, **fit_params)
print("Done!")


# Now we plot the loss and accuracy history of the model.

# In[ ]:


train_acc = history.history["acc"]
val_acc = history.history["val_acc"]
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.plot(train_acc)
plt.plot(val_acc)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(["train_acc", "val_acc"], loc="upper right")
plt.show()


# In[ ]:


plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train_loss", "val_loss"], loc="upper right")
plt.show()


# ## 4.2 Evaluate the Model
# Now we need to evaluate our trained model on the validation data.

# In[ ]:


loss, acc = model.evaluate(x_dev, y_dev)
print("Validation loss: {:.4f}".format(loss))
print("Validation accuracy: {:.4f}".format(acc))


# On the validation set our model reached an accuracy over 99%, which is pretty good. Now let's see some sample prediction:

# In[ ]:


num_samples = 10
dev_size = x_dev.shape[0]
sample_idx = np.random.randint(dev_size, size=num_samples)
x_samples = x_dev[sample_idx, :, :, :]
y_samples_pred = np.argmax(model.predict(x_samples), axis=1)

plt.figure(figsize=(20, 10))
for i in range(num_samples):
    
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_samples[i, :, :, 0], cmap="gray")
    plt.title("Prediction: {}".format(y_samples_pred[i]))


# Now let's plot the confusion matrix to see where we make mistakes most often.

# In[ ]:


y_val_true = np.argmax(y_dev, axis=1)
y_val_pred = np.argmax(model.predict(x_dev), axis=1)
conf_matrix = pd.DataFrame(confusion_matrix(y_val_true, y_val_pred), index=[i for i in range(10)], columns=[i for i in range(10)])
plt.figure(figsize = (10,7))
sbn.heatmap(conf_matrix, annot=True)


# We can see that the model sometimes confuse 1 or 2 with 7, and 4 with 9.

# # 5 Output the Prediction

# Now we output the predictions and save it in a .csv file.

# In[ ]:


y_pred = model.predict(x_test, batch_size=batch_size)
y_pred = np.argmax(y_pred, axis=1).reshape((-1, 1))
idx = np.reshape(np.arange(1, len(y_pred) + 1), (len(y_pred), -1))
y_pred = np.hstack((idx, y_pred))
y_pred = pd.DataFrame(y_pred, columns=['ImageId', 'Label'])
y_pred.to_csv('y_pred.csv', index=False)

