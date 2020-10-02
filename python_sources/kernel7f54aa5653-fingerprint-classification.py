#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy
import glob
import pylab as plt



get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math


# In[ ]:



# Use TensorFlow v.2 with this old v.1 code.
# E.g. placeholder variables and sessions have changed in TF2.
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[ ]:


tf.__version__


# In[ ]:


#convolution layer 1
filter_size1=3 # convolutional filters are 3x3 pixels
num_filters1= 128 #there are 16 of these filters
#convolutional layer 2
filter_size2=3 # convolutional filters are 3x3 pixels
num_filters2= 128 #there are 16 of these filters

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.


# In[ ]:


#fully connected layer
fc_size=128 #number of neurns in fully connected layer
num_channels=3
img_size =128
img_size_flat= img_size*img_size*num_channels

img_shape=[img_size,img_size]
epoch=500
batch=20
j=0


# In[ ]:


folders = glob.glob('../input/esther1/DB1_B/*')


# In[ ]:


len(folders)


# In[ ]:


imagenames_list = []
labels =[]
imagenames_list=[]
count = 0


# In[ ]:


for folder in folders:
    for f in glob.glob(folder+'/*.tif'):
        imagenames_list.append(f)
        labels.append(count)
    count+=1
read_images=[]
Tensor_input=[]


# In[ ]:


for image in imagenames_list:
    read_images.append(cv2.imread(image,cv2.IMREAD_GRAYSCALE))


# In[ ]:


labels=tf.keras.utils.to_categorical(
                 labels,
                  num_classes=3
                 )


# In[ ]:


#def new_weights(shape):
  #  return tf.Variable(tf.random.truncated_normal(shape, stddev=0.05))

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


# In[ ]:


#def new_biases(length):
 #   return tf.Variable(tf.constant(0.05, shape=[length]))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length])) 


# In[ ]:


def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights


# In[ ]:


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


# In[ ]:


def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


# In[ ]:


#tensor inputs
x= tf.compat.v1.placeholder(tf.float32,shape=[None,img_size_flat],name='x')


# In[ ]:


x_image = tf.reshape(x,[-1,img_size,img_size,num_channels])


# In[ ]:


y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')


# In[ ]:


y_true_cls = tf.argmax(y_true, axis=1)


# In[ ]:


#Convolutional Layer 1
layer_conv1,weights_conv1 =new_conv_layer(input=x_image,
                                          num_input_channels=num_channels,
                                          num_filters=num_filters1,
                                          filter_size=filter_size1,
                                          use_pooling=True
                                          )


# In[ ]:


layer_conv1


# In[ ]:


#Convolutional Layer 2
layer_conv2, weights_conv2 =     new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)


# In[ ]:


layer_conv2


# In[ ]:


#Flatten Layer
layer_flat, num_features = flatten_layer(layer_conv2)


# In[ ]:


layer_flat


# In[ ]:


num_features


# In[ ]:



#Fully-Connected Layer 1
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)


# In[ ]:


layer_fc1


# In[ ]:


layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=10,
                         use_relu=False)


# In[ ]:


layer_fc2


# In[ ]:


#Predicted Class

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)


# In[ ]:


#Cost-function to be optimized
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)


# In[ ]:


summary_d_loss =tf.summary.scalar('cross_entropy',cross_entropy)
cost = tf.reduce_mean(cross_entropy)
summary_COST_loss =tf.summary.scalar('cost', cost)


# In[ ]:


cost = tf.reduce_mean(cross_entropy)


# In[ ]:


#Optimization Method
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(cost)


# In[ ]:


#Performance Measures
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[ ]:


init = tf.global_variables_initializer()


# In[ ]:


# Initializing the variables
#init = tf.initialize_all_variables()
init=tf.compat.v1.global_variables_initializer()


# In[ ]:


with tf.Session() as sess:
   sess.run(init)
   saver=tf.train.Saver()
   merged =tf.summary.merge_all()
   writer = tf.summary.FileWriter('./logs',sess.graph)
   for i in range(1,epoch):
        batch_ss =read_images[j:batch+j]
        labels_ss =labels[j:batch+j]

        j+=batch
        if j+batch>=len(read_images):
            j=0

        feed_dict_train = {x:batch_ss,
                           y_true:labels_ss}

        summary_loss,loss=sess.run([summary_COST_loss,cost],feed_dict=feed_dict_train)
        

