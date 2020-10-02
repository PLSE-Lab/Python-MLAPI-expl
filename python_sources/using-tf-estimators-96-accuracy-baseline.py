#!/usr/bin/env python
# coding: utf-8

# # **Digit Classifier using Estimators**
# 
# Estimator class is used to train and evaluate TensorFlow models. It wraps a model which is specified by a model_fn, which, given inputs and a number of other parameters, returns the operations necessary to perform training, evaluation, and predictions.
# 
# We will use DNNClassifier to classify the hand-written digits in the MNIST dataset.
# 
# Start by importing the required packages we need to build the model

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

from __future__ import absolute_import, division, print_function
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)


# ## **Data Preparation and Visualization**
# 
# Load the training and test data

# In[ ]:


# Load the training and test data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


# separate out label into a different data frames
train_y = train['label']
train_x = train.drop(labels = ["label"],axis = 1)


# In[ ]:


# print the number of labels 
train_y.value_counts()


# This shows that we have approximately same number of training data for each digit. 

# ### **Normalization**
# 
# Normalization is a process that changes the range of pixel intensity values. The motivation is to achieve consistency in dynamic range for a set of data, signals, or images to avoid mental distraction.

# In[ ]:


# Normalize the data
train_x = train_x / 255.0
test = test / 255.0


# In[ ]:


# Reshape image in 3 dimensions (28 height x 28 width x 1 channel for gray)
train_x = train_x.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)


# ### **Augmentation**
# 
# We will use Data Augmentation technique to increase the training data set to avoid overfitting the model. The motivation is to generate variations of the training dataset (for eg, zooming the image, or shiting the number of the left/right, or rotating the image in any random direction)****

# In[ ]:


datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            zca_whitening=False,
                            width_shift_range = 0.1,
                            rotation_range = 10)

# this is not being used for now
datagen.fit(train_x)


# ## **Building the CNN Classifier**
# 
# CNN (Convolutional Neural Network) is the state of the art model whenever it comes to classifying images. The layers in CNN learns the high level features of the given image which is later used for classification.
# 
# The CNN consists of three major components:
# 
# 1. Convolutional Layer: Applies the convolutional filter to a given image. This means it applies mathematical operations for a sub region to produce a single value in a feature output map.
# 2. Pooling Layer: It is used to reduce the dimensionality of the image to improve the training time.
# 3. Dense Layer: This is the final layer which performs the classification.
# 
# Since now we have a overall idea of what CNN is, let's try building a CNN model using the above mentioned layers

# In[ ]:


def cnn_model(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # 1st Convolutional Layer
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, 
                           kernel_size=[5, 5], padding="same",
                           activation=tf.nn.relu)

    # 1st Pooling Layer
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # 2nd Convolutional Layer
    # This takes the output of previous pool layer as it's input
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, 
                           kernel_size=[5, 5],padding="same", 
                           activation=tf.nn.relu)
    
    # 2nd Pooling Layer
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Before connecting the layer, we'll flatten the feature map
    dense = tf.layers.dense(inputs=tf.reshape(pool2, [-1, 7 * 7 * 64]), units=1024, activation=tf.nn.relu)
    
    # to improve the results apply dropout regularization to the layer to reduce overfitting
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# ### **Breaking it down**
# 
# This is how the model looks like: **Conv Layer -> Pool Layer -> Conv Layer -> Pool Layer -> Flatten -> Dropout Layer -> Dense Output Layer**
# 
# **Input Layer**
# 
# > input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
# 
# The Convolutional and pooling layer expects the inputs to be in format of [batch_size, image_height, image_width, channels]. Since we have 28(width) x 28(height) x 1(channel) image dataset, hence the shape of the input layer.
# 
# **1st Convolutional Layer**
# 
# > conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, 
# >                            kernel_size=[5, 5], padding="same",
# >                            activation=tf.nn.relu)
# 
# The first convolutional layer applies 32 5x5 filters to the input layer with ReLU as the activation function (to learn non-learnier features)
# 
# **1st Pooling Layer**
# 
# > pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
# 
# The pooling layers applies a pool size of 2 to the output of the 1st convolutional layer to reduce the dimensionality of the inputs. The strides argument specifies the size of the stride. Stride of 2 means that the subregions extracted by the filter should be separated by 2 pixels.
# 
# **2nd Convolutional Layer**
# 
# > conv2 = tf.layers.conv2d(inputs=pool1, filters=64, 
# >                            kernel_size=[5, 5],padding="same", 
# >                            activation=tf.nn.relu)
# 
# The convolutional layer of the same parameters are applied to the output of the 1st pooling layer.
# 
# **2nd Pooling Layer**
# 
# > pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
# 
# The pooling layer of the same parameters are applied to the output of the 2nd convolutional layer.
# 
# **Flattent Layer**
# 
# > dense = tf.layers.dense(inputs=tf.reshape(pool2, [-1, 7 * 7 * 64]), units=1024, activation=tf.nn.relu)
# 
# Before generating the dense layer, we will have to flatten the feature map
# 
# **Dropout Layer**
# 
# > dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
# 
# To help improve the results of the model we  apply dropout regularization to our dense flattened layer. The rate argument specifies the dropout rate; here, we use 0.4, which means 40% of the elements will be randomly dropped out during the training phase.
# 
# **Dense Logits Layer**
# 
# > logits = tf.layers.dense(inputs=dropout, units=10)
# 
# The final layer is the logits layer, which will return the raw values for the predictions.

# In[ ]:


import numpy as np
import tensorflow as tf
import logging
from tensorflow.python.training import session_run_hook

class EarlyStoppingHook(session_run_hook.SessionRunHook):
    """Hook that requests stop at a specified step."""

    def __init__(self, monitor='val_loss', min_delta=0, patience=0,
                 mode='auto'):
        """
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        if mode not in ['auto', 'min', 'max']:
            logging.warning('EarlyStopping mode %s is unknown, '
                            'fallback to auto mode.', mode, RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def begin(self):
        # Convert names to tensors if given
        graph = tf.get_default_graph()
        self.monitor = graph.as_graph_element(self.monitor)
        if isinstance(self.monitor, tf.Operation):
            self.monitor = self.monitor.outputs[0]

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return session_run_hook.SessionRunArgs(self.monitor)

    def after_run(self, run_context, run_values):
        current = run_values.results

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                run_context.request_stop()


# ### **Training the classifier**
# 
# Since we have the model ready, let's train the model using the Estimator API's

# In[ ]:


# Create the Estimator
classifier = tf.estimator.Estimator(model_fn=cnn_model, model_dir="/tmp/model")

early_stopping_hook = EarlyStoppingHook(monitor='sparse_softmax_cross_entropy_loss/value', patience=10)

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_x},
                                                    y=train_y,
                                                    batch_size=64,
                                                    num_epochs=200,
                                                    shuffle=True)

classifier.train(input_fn=train_input_fn, steps=30000)


# ### **Predicting values**

# In[ ]:


predict_fn = tf.estimator.inputs.numpy_input_fn(x={"x": test},
                                                shuffle=False)

eval_results = classifier.predict(input_fn=predict_fn)


# In[ ]:


# predict results
result = []
for i in eval_results:
    result.append(i['classes'])

results = pd.Series(result,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submission.csv",index=False)


# ## **Pending Tasks**
# 
# This model is a baseline model. There is a lot of scope of impovement here like tuning the hyperparameters, trying out different types of learning optimizers, and more
# 
# 1. Hyperparameter tuning: Tuning the learning rate, with different optimizers and loss functions
# 2. Data Augementation: Enable data augmentation to reduce overfitting
# 3. Adding early stop ie train till the model stops learning and thus auto stops
# 4. Defining the confusion matrix
# 5. Analysing ROC curve to evaluate the performance of classification

# In[ ]:




