#!/usr/bin/env python
# coding: utf-8

# Use Resnet model for bottleneck feature extraction.
# ---------------------------
# 
# Resnet pre-trained model is available [here][1].
# I have used Inception_resnet_v2 model.
# 
# If you come across this transfer learning concept first time, go through the [chapter of stanford CS231][2]. This blog will clear all your doubts about transfer learning.
# 
# 
#   [1]: https://github.com/tensorflow/models/tree/master/slim
#   [2]: http://cs231n.github.io/transfer-learning/

# Import required libraries

# In[ ]:


import numpy as np
import pandas as pd
import cv2
import math
from glob import glob
import os
from scipy.misc import imread, imresize, imshow
from sklearn.model_selection import train_test_split
import tensorflow.contrib.slim as slim
import tensorflow as tf
from sklearn.linear_model import LogisticRegression


# In[ ]:


master = pd.read_csv("../input/train_labels.csv")
master.head()


# Function read images in batches.

# In[ ]:


data_dir = "../input/train/"

def get_image_data_for_batch(images_in_batch):
    batch_image_data = []#np.empty([10, 299,299,3], dtype=np.int)
    for ix in images_in_batch:
        temp = imread(data_dir+str(ix)+".jpg")
        temp = imresize(temp, [299,299,3])
        batch_image_data.append(temp)# - np.mean(temp)
    return np.array(batch_image_data)


# Let's declare the whole inception_resnet model

# In[ ]:



import tensorflow as tf

slim = tf.contrib.slim


def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 35x35 resnet block."""
  with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
    with tf.variable_scope('Branch_2'):
      tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
      tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net


def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 17x17 resnet block."""
  with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7],
                                  scope='Conv2d_0b_1x7')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1],
                                  scope='Conv2d_0c_7x1')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net


def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 8x8 resnet block."""
  with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],
                                  scope='Conv2d_0b_1x3')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],
                                  scope='Conv2d_0c_3x1')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net


def inception_resnet_v2(inputs, num_classes=1001, is_training=True,
                        dropout_keep_prob=0.8,
                        reuse=None,
                        scope='InceptionResnetV2'):
  """Creates the Inception Resnet V2 model.

  Args:
    inputs: a 4-D tensor of size [batch_size, height, width, 3].
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    dropout_keep_prob: float, the fraction to keep before final layer.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    logits: the logits outputs of the model.
    end_points: the set of end_points from the inception model.
  """
  end_points = {}

  with tf.variable_scope(scope, 'InceptionResnetV2', [inputs], reuse=reuse):
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                          stride=1, padding='SAME'):

        # 149 x 149 x 32
        net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID',
                          scope='Conv2d_1a_3x3')
        end_points['Conv2d_1a_3x3'] = net
        # 147 x 147 x 32
        net = slim.conv2d(net, 32, 3, padding='VALID',
                          scope='Conv2d_2a_3x3')
        end_points['Conv2d_2a_3x3'] = net
        # 147 x 147 x 64
        net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
        end_points['Conv2d_2b_3x3'] = net
        # 73 x 73 x 64
        net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                              scope='MaxPool_3a_3x3')
        end_points['MaxPool_3a_3x3'] = net
        # 73 x 73 x 80
        net = slim.conv2d(net, 80, 1, padding='VALID',
                          scope='Conv2d_3b_1x1')
        end_points['Conv2d_3b_1x1'] = net
        # 71 x 71 x 192
        net = slim.conv2d(net, 192, 3, padding='VALID',
                          scope='Conv2d_4a_3x3')
        end_points['Conv2d_4a_3x3'] = net
        # 35 x 35 x 192
        net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                              scope='MaxPool_5a_3x3')
        end_points['MaxPool_5a_3x3'] = net

        # 35 x 35 x 320
        with tf.variable_scope('Mixed_5b'):
          with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
          with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,
                                        scope='Conv2d_0b_5x5')
          with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,
                                        scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,
                                        scope='Conv2d_0c_3x3')
          with tf.variable_scope('Branch_3'):
            tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME',
                                         scope='AvgPool_0a_3x3')
            tower_pool_1 = slim.conv2d(tower_pool, 64, 1,
                                       scope='Conv2d_0b_1x1')
          net = tf.concat(axis=3, values=[tower_conv, tower_conv1_1,
                              tower_conv2_2, tower_pool_1])

        end_points['Mixed_5b'] = net
        net = slim.repeat(net, 10, block35, scale=0.17)

        # 17 x 17 x 1088
        with tf.variable_scope('Mixed_6a'):
          with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 384, 3, stride=2, padding='VALID',
                                     scope='Conv2d_1a_3x3')
          with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,
                                        scope='Conv2d_0b_3x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,
                                        stride=2, padding='VALID',
                                        scope='Conv2d_1a_3x3')
          with tf.variable_scope('Branch_2'):
            tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                         scope='MaxPool_1a_3x3')
          net = tf.concat(axis=3, values=[tower_conv, tower_conv1_2, tower_pool])

        end_points['Mixed_6a'] = net
        net = slim.repeat(net, 20, block17, scale=0.10)

        # Auxiliary tower
        with tf.variable_scope('AuxLogits'):
          aux = slim.avg_pool2d(net, 5, stride=3, padding='VALID',
                                scope='Conv2d_1a_3x3')
          aux = slim.conv2d(aux, 128, 1, scope='Conv2d_1b_1x1')
          aux = slim.conv2d(aux, 768, aux.get_shape()[1:3],
                            padding='VALID', scope='Conv2d_2a_5x5')
          aux = slim.flatten(aux)
          aux = slim.fully_connected(aux, num_classes, activation_fn=None,
                                     scope='Logits')
          end_points['AuxLogits'] = aux

        with tf.variable_scope('Mixed_7a'):
          with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
            tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                       padding='VALID', scope='Conv2d_1a_3x3')
          with tf.variable_scope('Branch_1'):
            tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
                                        padding='VALID', scope='Conv2d_1a_3x3')
          with tf.variable_scope('Branch_2'):
            tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
                                        scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
                                        padding='VALID', scope='Conv2d_1a_3x3')
          with tf.variable_scope('Branch_3'):
            tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                         scope='MaxPool_1a_3x3')
          net = tf.concat(axis=3, values=[tower_conv_1, tower_conv1_1,
                              tower_conv2_2, tower_pool])

        end_points['Mixed_7a'] = net

        net = slim.repeat(net, 9, block8, scale=0.20)
        net = block8(net, activation_fn=None)

        net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
        end_points['Conv2d_7b_1x1'] = net

        with tf.variable_scope('Logits'):
          end_points['PrePool'] = net
          net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                scope='AvgPool_1a_8x8')
          net = slim.flatten(net)

          net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                             scope='Dropout')

          end_points['PreLogitsFlatten'] = net
          logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                        scope='Logits')
          end_points['Logits'] = logits
          end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')

    return logits, end_points
inception_resnet_v2.default_image_size = 299


def inception_resnet_v2_arg_scope(weight_decay=0.00004,
                                  batch_norm_decay=0.9997,
                                  batch_norm_epsilon=0.001):
  """Yields the scope with the default parameters for inception_resnet_v2.

  Args:
    weight_decay: the weight decay for weights variables.
    batch_norm_decay: decay for the moving average of batch_norm momentums.
    batch_norm_epsilon: small float added to variance to avoid dividing by zero.

  Returns:
    a arg_scope with the parameters needed for inception_resnet_v2.
  """
  # Set weight_decay for weights in conv2d and fully_connected layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_regularizer=slim.l2_regularizer(weight_decay)):

    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
    }
    # Set activation_fn and parameters for batch_norm.
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params) as scope:
      return scope


# Now we can extract bottleneck features from any layer. I am using 2nd last layer of the network.

# In[ ]:


X = tf.placeholder(tf.float32, shape=[None, 299,299,3])
with slim.arg_scope(inception_resnet_v2_arg_scope()):
    logits, end_points = inception_resnet_v2(X, num_classes=1001,is_training=False)
predictions = end_points["PreLogitsFlatten"] #you can try with other layers too. Intermediate layers 
#may give good accuracy, but feature size for each image will be in 10k's.


# You have to download the checkpoint for initializing the network with pre-trained weights. Link to download the checkpoint file is [here][1].  As I am unable load this checkpoint file here, further part of this notebook won't run here. But I have tested it on my machine, works well.
# 
# 
#   [1]: http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz

# In[ ]:


X = tf.placeholder(tf.float32, shape=[None, 299,299,3])
with slim.arg_scope(inception_resnet_v2_arg_scope()):
    logits, end_points = inception_resnet_v2(X, num_classes=1001,is_training=False)
predictions = end_points["PreLogitsFlatten"]
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, "PATH TO inception_resnet_v2_2016_08_30.ckpt")


# Now we can extract features from pre-trained model and use those features for applying any other algorithm.

# In[ ]:


number_of_batches = int(np.ceil(float(len(master))/10))
bottleneck_features_train = np.empty(shape=[0, 1536])
for i in range(number_of_batches):
    batch_x = master[(i*10):(i*10)+10]['name']
    image_data = get_image_data_for_batch(batch_x)
    btnk_batch = sess.run(predictions, feed_dict={X:image_data})
    bottleneck_features_train = np.concatenate([bottleneck_features_train, btnk_batch], axis=0)


# Same way, we have extract features for test images.

# In[ ]:


data_dir = "../input/test/"
test_image_names = range(1, 1532)
number_of_batches_test = int(np.ceil(1531.0/10))
bottleneck_features_test = np.empty(shape=[0, 1536])
for i in range(number_of_batches_test):
    batch_x = test_image_names[(i*10):(i*10)+10]
    image_data = get_image_data_for_batch(batch_x)
    btnk_batch = sess.run(predictions, feed_dict={X:image_data})
    bottleneck_features_test = np.concatenate([bottleneck_features_test, btnk_batch], axis=0)


# In[ ]:


I have used Logistic regression from scikit learn, you can use other powerful algorithms like randomForest, xgboost etc.


# In[ ]:


model =  LogisticRegression()
model.fit(bottleneck_features_train,master['invasive'])

predictions = model.predict(bottleneck_features_test)
#Read sample submission file 
sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission['invasive'] = predictions

sample_submission.to_csv("inception_resnet_logistic", index=False)


# Next task will be to try different layers for bottleneck extraction, try other networks like InceptionV4, Resnet-150. Also to fine-tune these networks with our data.
