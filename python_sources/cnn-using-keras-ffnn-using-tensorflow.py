#!/usr/bin/env python
# coding: utf-8

# # CNN using Keras & FFNN using tensorflow
# #### Mahdi Shadkam-Farrokhi: [GitHub](https://github.com/Shaddyjr) | [Medium](https://medium.com/@mahdis.pw) | [LinkedIn](https://www.linkedin.com/in/mahdi-shadkam-farrokhi-m-s-8a410958/) | [mahdis.pw](http://mahdis.pw)
# 
# ---

# ## Table of Contents
# 
# - [Using Keras w/CNN](#Using-Keras-w/CNN)
#     - [Loading the data](#CNN:-Loading-the-data)
#     - [Cleaning the data](#CNN:-Cleaning-the-data)
#     - [EDA](#EDA)
#     - [Model Preparation](#CNN:-Model-Preparation)
#     - [Model Experimentation](#CNN:-Model-Experimentation)
#     - [Model Evaluation](#CNN:-Model-Evaluation)
#     - [Kaggle Submission](#CNN:-Kaggle-Submission)
# - [Using Tensorflow w/network](#Using-Tensorflow-w/network)
#     - [Loading the data](#Tf:-Loading-the-data)
#     - [Cleaning the data](#Tf:-Cleaning-the-data)
#     - [Model Preparation](#Tf:-Model-Preparation)
#     - [Model Experimentation](#Tf:-Model-Experimentation)
#     - [Model Evaluation](#Tf:-Model-Evaluation)
#     - [Kaggle Submission](#Tf:-Kaggle-Submission)
# - [Sources](#Sources)

# In[ ]:


# Import libraries and modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# For reproducibility
np.random.seed(42)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils, to_categorical
from keras.optimizers import Adam

import tensorflow as tf

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# ## Using Keras w/CNN
# [Back to Table of Contents](#Table-of-Contents)

# ### CNN: Loading the data

# In[ ]:


import os
os.listdir("../input/digit-recognizer")


# Data files are in the "../input/digit-recognizer" directory

# In[ ]:


DATA_PATH = "../input/digit-recognizer/"
digits = pd.read_csv(DATA_PATH + "train.csv")


# In[ ]:


digits.head()


# These data are in a 1D format, which will need to be converted to a proper 2D image format (with an alpha channel).

# In[ ]:


digits.shape


# ### CNN: Cleaning the data

# Image data works best in CNN when scaled and normalized.
# 
# [source](https://www.jeremyjordan.me/batch-normalization/)

# In[ ]:


target = np.array(digits["label"])
data = digits.drop(columns = "label")


# In[ ]:


def transform_raw_data(data):
    '''returns the scaled and normalized data reshaped to proper 2D format'''
    data = np.array(data)
    shape = (data.shape[0], 28,28, 1)
    X = data.reshape(shape)
    X = X / 255 # rescale
    X = (X - .5).astype("float16")  # normalize (optimization for CNN)
    return X


# In[ ]:


X = transform_raw_data(data)


# In[ ]:


X.shape


# ### EDA
# Looking at the data, we can gain a better understanding of the underlying nature of the data.

# In[ ]:


def show_image(image, ax = plt, title = None):
    '''displays a single image using a given axes'''
    ax.imshow(image.reshape((28,28)), cmap="gray")
    if title:
        ax.set_title(title)
    ax.tick_params(bottom = False, left = False, labelbottom = False, labelleft = False)


# In[ ]:


def show_images(images, titles = None, ncols = 4, height = 2):
    '''displays a list/array of images in a grid format'''
    nrows = int(np.ceil(len(images)/ncols))
    f, ax = plt.subplots(nrows=nrows,ncols=ncols, figsize=(10,nrows * height))
    ax = ax.flatten()
    for i, image in enumerate(images):
        if titles:
            show_image(image, ax = ax[i], title = titles[i])
        else:
            show_image(image, ax = ax[i], title = None)
    plt.tight_layout()
    plt.show()


# In[ ]:


show_images(np.asarray(data[:4]).reshape((4,28,28)))


# Here, we can immediately see the diversity found within the human-written dataset.

# ### CNN: Model Preparation

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, target, stratify = target, test_size=.1)


# In[ ]:


y_train_cat = np_utils.to_categorical(y_train, num_classes=10)
y_test_cat = np_utils.to_categorical(y_test, num_classes=10)


# ### CNN: Model Experimentation

# In[ ]:


def graph_loss(history):
    '''graphs training and testing loss given a keras History object'''
    # Check out our train loss and test loss over epochs.
    train_loss = history.history['loss']
    test_loss = history.history['val_loss']
    xticks = np.array(range(len(train_loss)))
    # Set figure size.
    plt.figure(figsize=(12, 8))

    # Generate line plot of training, testing loss over epochs.
    plt.plot(train_loss, label='Training Loss', color='#185fad')
    plt.plot(test_loss, label='Testing Loss', color='orange')

    # Set title
    plt.title('Training and Testing Loss by Epoch', fontsize = 25)
    plt.xlabel('Epoch', fontsize = 18)
    plt.ylabel('Categorical Crossentropy', fontsize = 18)
    plt.xticks(xticks, xticks+1)

    plt.legend(fontsize = 18);


# For this model, I experimented with various structures, and the $[(2 conv + 1 pool) * 2 + 2 Dense]$ model worked the best, give the accuracy metric.

# In[ ]:


# setup model
model = Sequential([

    Conv2D(32, input_shape = (28,28, 1), kernel_size = 5, activation="relu", padding="same"),
    Conv2D(32, kernel_size = 5, activation="relu", padding = 'same'),
    MaxPooling2D((2,2)),
    Dropout(.25),
    
    Conv2D(64, kernel_size = 3, activation="relu", padding = 'same'),
    Conv2D(64, kernel_size = 3, activation="relu", padding = 'same'),
    MaxPooling2D((2,2), strides=(2,2)),
    Dropout(.25),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(.5),
    
    Dense(64, activation="relu"),
    Dropout(.5),

    Dense(10, activation="softmax"),
])


# In[ ]:


# compile model
model.compile(
    loss = "categorical_crossentropy",
    optimizer = "adam", # Adam(lr = .0001, decay= 1e-5),
    metrics = ["acc"]
)


# In[ ]:


# fit model
history = model.fit(
    X_train,
    y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs = 50,
    batch_size= 64
)


# In[ ]:


graph_loss(history)


# We see a steady decrease in the traning loss, and an unusually low and constant testing loss.

# ### CNN: Model Evaluation

# In[ ]:


preds = model.predict_classes(X_test)


# In[ ]:


results = pd.DataFrame({"actual":y_test,"pred":preds,"is_correct":y_test == preds})
errors = results[results["is_correct"] == False]
errors.head()


# In[ ]:


results["is_correct"].value_counts(normalize=True)


# #### Evaluating model erros

# In[ ]:


titles = ["Actual:{}\nPred:{}".format(act,errors["pred"].iloc[i]) for i,act in enumerate(errors["actual"][:12])]


# In[ ]:


show_images(((X_test + .5) * 255)[errors.index][:12].astype(int), titles = titles, ncols = 6, height=3)


# The errors the model makes are quite understandable. Many of them are not even human-readable!

# ### CNN: Kaggle Submission

# In[ ]:


test = pd.read_csv(DATA_PATH + "test.csv")


# In[ ]:


test.head()


# In[ ]:


kaggle_X = transform_raw_data(test)

kaggle_X.shape


# In[ ]:


kaggle_preds = model.predict_classes(kaggle_X)


# In[ ]:


final = pd.DataFrame({"ImageId":test.index + 1,"Label":kaggle_preds})

final.head()


# In[ ]:


final.to_csv("submission.csv", index = False)


# ## Using Tensorflow w/network
# 
# Tensorflow will be used to address the problem with a Feed-Forward Neural Network.  
# 
# [Back to Table of Contents](#Table-of-Contents)

# ### Tf: Loading the data

# In[ ]:


digits = pd.read_csv(DATA_PATH + "train.csv")


# In[ ]:


digits.head()


# The data are in a 1D format, but would work best as a `np.array`. We would also like to scale and normalize according to best practices for running a model through a NN.

# ### Tf: Cleaning the data

# In[ ]:


data = np.asarray(digits.drop(columns = "label"))
target = to_categorical(digits["label"], num_classes = 10) # turns into matrix for us!


# The data is already flattened, so no need for too much modification.

# In[ ]:


data.shape


# In[ ]:


target.shape


# In[ ]:


def transform_raw_data_TF(data):    
    # rescales and normalizes (optimization for FFNN)
    return (data / 255 - .5).astype("float32")


# In[ ]:


data = transform_raw_data_TF(data)


# Similar to how the data was transformed in the CNN, these data will be scaled and normalized for the Feed Forward Neural Network.

# ### Tf: Model Preparation

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data, target, random_state = 42)


# ### Tf: Model Experimentation
# 
# This FFNN features the following layers:  
# 
# $Dense 128 + Dropout + Dense 64 + Dropout$
# 
# using ReLU and an Adam optimizer

# In[ ]:


tf.reset_default_graph()

### START OF NETWORK ###
X_scaffold = tf.placeholder(dtype = tf.float32, shape = (None, X_train.shape[1]))
y_scaffold = tf.placeholder(dtype = tf.float32, shape = (None, y_train.shape[1]))

h1 = tf.layers.dense(X_scaffold, 128, activation = tf.nn.relu)

# DROPOUT CAUSING DIFFERENT TEST OUTPUT, NEED TO TURN OFF DURING INFERENCING
prob = tf.placeholder_with_default(0.0, shape=())

d1 = tf.layers.dropout(h1, rate = prob)
h2 = tf.layers.dense(d1, 64, activation = tf.nn.relu)
d2 = tf.layers.dropout(h2, rate = prob)

y_hat = tf.layers.dense(d2, y_train.shape[1], activation = None)

ouput = tf.nn.softmax(y_hat)
### END OF NETWORK ###

loss = tf.losses.softmax_cross_entropy(y_scaffold, y_hat)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
training_epoch = optimizer.minimize(loss)

saver = tf.train.Saver()


# In[ ]:


init = tf.global_variables_initializer()
n_epochs = 500

sess = tf.Session()
with sess:
    sess.run(init)
    train_loss = []
    test_loss  = []
    for epoch in range(n_epochs):
        sess.run(training_epoch, feed_dict={
            X_scaffold : X_train, 
            y_scaffold : y_train,
            prob       : .1     # DROPOUT CAUSING DIFFERENT TEST OUTPUT, NEED TO TURN OFF DURING INFERENCING
        }) 
        train_loss.append(sess.run(loss, feed_dict={X_scaffold : X_train, y_scaffold : y_train}))
        test_loss.append(sess.run(loss, feed_dict={X_scaffold: X_test, y_scaffold: y_test}))
    pred = sess.run(y_hat, feed_dict={X_scaffold: X_test})
    saver.save(sess, './sess.ckpt')   


# ### Tf: Model Evaluation

# In[ ]:


def graph_loss(train_loss, test_loss):
    '''Graphing function to visualize loss'''
    xticks = np.array(range(len(train_loss)))
    # Set figure size.
    plt.figure(figsize=(12, 8))

    # Generate line plot of training, testing loss over epochs.
    plt.plot(train_loss, label='Training Loss', color='#185fad')
    plt.plot(test_loss, label='Testing Loss', color='orange')

    # Set title
    plt.title('Training and Testing Loss by Epoch', fontsize = 25)
    plt.xlabel('Epoch', fontsize = 18)
    plt.ylabel('Categorical Crossentropy', fontsize = 18)
    plt.xticks(xticks[::10], (xticks+1)[::10])

    plt.legend(fontsize = 18);


# In[ ]:


graph_loss(train_loss, test_loss)


# Here we see the classic signs of overfitting over time. With each new epoch, the variance between the training and testing dataset gets larger. However, the difference is small.

# In[ ]:


y_pred = pred.argmax(axis = 1)
y_true = y_test.argmax(axis = 1)
cm = confusion_matrix(y_true, y_pred)
pd.DataFrame(cm,
             index=["Actual {}".format(num+1) for num in range(10)],
             columns=["Pred. {}".format(num+1) for num in range(10)])


# In[ ]:


np.mean(y_true == y_pred)


# This model has a respectable 96.58% accurary. Although lower than the CNN accuracy, this model would likely be improved by using batching, increasing the number of nodes, and/or increasing the number of epochs.

# ### Tf: Kaggle Submission

# In[ ]:


test = pd.read_csv(DATA_PATH + "test.csv")


# In[ ]:


test.head()


# In[ ]:


kaggle_X = transform_raw_data_TF(np.asarray(test))

kaggle_X.shape


# In[ ]:


with tf.Session() as sess:
    saver.restore(sess, './sess.ckpt')
    kaggle_preds = sess.run(y_hat, feed_dict={X_scaffold: kaggle_X})


# In[ ]:


kaggle_preds_final = kaggle_preds.argmax(axis = 1)
kaggle_preds_final


# In[ ]:


# sneak peek at kaggle answers
show_images(np.asarray(test).reshape((-1, 28,28,1))[:8], titles = ["Pred: {}".format(i) for i in kaggle_preds_final[:8]])


# Inputting human-interpreted values is against the merit of this challenge.
# 
# These visualized test images were used as a verification check when model was giving starkly different predictions on repeated calls to `sess.run(y_hat, feed_dict={X_scaffold: kaggle_X})`. In the end, the issue was resolved using a stored session.

# In[ ]:


final = pd.DataFrame({"ImageId":test.index + 1,"Label":kaggle_preds_final})

final.head()


# In[ ]:


final.to_csv("submission_tf.csv", index = False)


# ## Sources

# - [Kaggle Digits Competition](https://www.kaggle.com/c/digit-recognizer)
