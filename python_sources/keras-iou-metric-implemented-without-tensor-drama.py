#!/usr/bin/env python
# coding: utf-8

# **Version 2:**
# 
# In the previous version I implemented a simple IoU function imagining a graph with the origin (0,0) located in the bottom left. In this version I've updated the IoU function imagining a numpy array with the origin (0,0) located at the top left corner. 
# 
# This version can be a bit confusing. To understand it better keep in mind that:
# 
# Given bounding box: *[x, y, width, height, conf_score]*
# 
# then index selection in numpy will take the form *array[y,x]* i.e. *array[row, column]*.

# #### Introduction

# In this kernel I will quickly demonstrate a method for implementing a custom metric in Keras without having to use tensors to write it. A tensorflow wrapper function called tf.py_func makes this possible. The body of the function is written using numpy and then inserted into py_func. You can think of py_func as a kind of Trojan horse.
# 
# We will implement a basic IoU (Intersection Over Union) metric.

# ### What is IoU?

# This is a good blog post that clearly explains IoU.
# 
# https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

# ### What is the form of y, the target?

# We will be predicting one bounding box. The output has 5 columns. This is the form:
# 
# y_true = [x, y, width, height, conf]<br>
# y_pred = [x, y, width, height, conf]
# 
# x and y are the coordinates of the top left corner of the box and<br>
# conf is the confidence score.

# ### What output do we get from Keras?

# As the model is running Keras outputs batches of y_pred and y_true as numpy arrays. So if y is as shown above and we have a batch size of 10, then y_pred and y_true will be numpy arrays with a shape (10,5) i.e. 10 rows and 5 colums.

# ### The IoU Function

# Our custon IoU function will take y_true and y_pred as inputs and output the average IoU score for the batch - as a scalar of type float32. This is what py_func needs to work its magic.
# Something to note is that all caculations done within the function must be numpy. For example np.min() and np.max(). Python functions should not be used.

# This is my simple IoU implementation:

# In[ ]:


def calculate_iou(y_true, y_pred):
    
    
    """
    Input:
    Keras provides the input as numpy arrays with shape (batch_size, num_columns).
    
    Arguments:
    y_true -- first box, numpy array with format [x, y, width, height, conf_score]
    y_pred -- second box, numpy array with format [x, y, width, height, conf_score]
    x any y are the coordinates of the top left corner of each box.
    
    Output: IoU of type float32. (This is a ratio. Max is 1. Min is 0.)
    
    """

    
    results = []
    
    for i in range(0,y_true.shape[0]):
    
        # set the types so we are sure what type we are using
        y_true = y_true.astype(np.float32)
        y_pred = y_pred.astype(np.float32)


        # boxTrue
        x_boxTrue_tleft = y_true[0,0]  # numpy index selection
        y_boxTrue_tleft = y_true[0,1]
        boxTrue_width = y_true[0,2]
        boxTrue_height = y_true[0,3]
        area_boxTrue = (boxTrue_width * boxTrue_height)

        # boxPred
        x_boxPred_tleft = y_pred[0,0]
        y_boxPred_tleft = y_pred[0,1]
        boxPred_width = y_pred[0,2]
        boxPred_height = y_pred[0,3]
        area_boxPred = (boxPred_width * boxPred_height)


        # calculate the bottom right coordinates for boxTrue and boxPred

        # boxTrue
        x_boxTrue_br = x_boxTrue_tleft + boxTrue_width
        y_boxTrue_br = y_boxTrue_tleft + boxTrue_height # Version 2 revision

        # boxPred
        x_boxPred_br = x_boxPred_tleft + boxPred_width
        y_boxPred_br = y_boxPred_tleft + boxPred_height # Version 2 revision


        # calculate the top left and bottom right coordinates for the intersection box, boxInt

        # boxInt - top left coords
        x_boxInt_tleft = np.max([x_boxTrue_tleft,x_boxPred_tleft])
        y_boxInt_tleft = np.max([y_boxTrue_tleft,y_boxPred_tleft]) # Version 2 revision

        # boxInt - bottom right coords
        x_boxInt_br = np.min([x_boxTrue_br,x_boxPred_br])
        y_boxInt_br = np.min([y_boxTrue_br,y_boxPred_br]) 

        # Calculate the area of boxInt, i.e. the area of the intersection 
        # between boxTrue and boxPred.
        # The np.max() function forces the intersection area to 0 if the boxes don't overlap.
        
        
        # Version 2 revision
        area_of_intersection =         np.max([0,(x_boxInt_br - x_boxInt_tleft)]) * np.max([0,(y_boxInt_br - y_boxInt_tleft)])

        iou = area_of_intersection / ((area_boxTrue + area_boxPred) - area_of_intersection)


        # This must match the type used in py_func
        iou = iou.astype(np.float32)
        
        # append the result to a list at the end of each loop
        results.append(iou)
    
    # return the mean IoU score for the batch
    return np.mean(results)


# ### Define a function containing py_func

# The input arguments will be y_true, y_pred and the "calculate_iou" function we created above.

# In[ ]:


def IoU(y_true, y_pred):
    
    # Note: the type float32 is very important. It must be the same type as the output from
    # the python function above or you too may spend many late night hours 
    # trying to debug and almost give up.
    
    iou = tf.py_func(calculate_iou, [y_true, y_pred], tf.float32)

    return iou


# ### How to use this with Keras?

# We need to simply enter the IoU function as a metric in the compile step, like this:

# In[ ]:


# model.compile(optimizer='Adam', loss='mse', metrics=[IoU])


# #### and that's it. 

# <hr>

# ### Testing the functions

# This is a quick test to make sure that the functions we created are working as expected. The final answer should be:<br> IoU = 0.153846

# In[ ]:


import numpy as np
import tensorflow as tf

# create two inputs simulating a batch_size = 3
# shape (3,5)
y_true = np.array([[1,5,2,3,0.5], [1,5,2,3,0.5], [1,5,2,3,0.5]])
y_pred = np.array([[2,4,3,3,0.7], [2,4,3,3,0.7], [2,4,3,3,0.7]])

# call the first function
result = calculate_iou(y_true, y_pred)

print(result)
print(type(result))


# In[ ]:


# call the second function
iou = IoU(y_true, y_pred)

print(type(iou))


# In[ ]:


# because iou is a tensor we can only print it inside a Tensorflow session

with tf.Session() as sess:
    print(sess.run(iou))


# <hr>

# ### Resources
# 
# These are some resources that I found helpful when learning this subject:

# Tensorflow info on py_func:<br>
#  https://www.tensorflow.org/api_docs/python/tf/py_func
# 
# Blog post explaining IoU:<br>
#  https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
# 
# Kernel by @aglotero where I first saw py_func used:<br>
# https://www.kaggle.com/aglotero/another-iou-metric
# 
# 

# <hr>
# 
# Thank you for reading.

# In[ ]:




