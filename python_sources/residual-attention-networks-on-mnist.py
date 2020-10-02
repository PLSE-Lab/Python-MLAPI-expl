#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
trainRaw=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
trainRaw=trainRaw.sample(frac=1)
trainRaw=trainRaw.reset_index(drop=True)
trainRaw.head()


# In[ ]:


train_labels=trainRaw['label'].tolist()
trainRaw=trainRaw.drop(columns=['label'],axis=1) 
trainRaw=np.array(trainRaw)


# In[ ]:


## Test of the training images:
plt.imshow(trainRaw[0].reshape(28,28))


# In[ ]:


train_xdata=np.array([x.reshape(28,28,1) for x in trainRaw])
## Sanity check
print(train_xdata.shape)


# In[ ]:


testRaw=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
testRaw.head()


# In[ ]:


testRaw=np.array(testRaw)


# In[ ]:


plt.imshow(testRaw[0].reshape(28,28))


# In[ ]:


test_xdata=np.array([x.reshape(28,28,1) for x in testRaw])
## Sanity check
print(test_xdata.shape)


# In[ ]:


from tensorflow.python.framework import ops
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
ops.reset_default_graph()
def residualAttentionBlock(x):
    input_shape=x.shape
    
    input_height=int(input_shape[1])
    input_width=int(input_shape[2])
    input_channels=int(input_shape[3])
    
    ## Hourglass part -- Convolutional Layer1
    conv1AR=tf.layers.conv2d(inputs=x,filters=input_channels*4,
                             kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
    max_pool1AR = tf.layers.max_pooling2d(inputs=conv1AR, pool_size=[2, 2], strides=2)
    
    ## Hourglass part -- Convolutional Layer2
    conv2AR=tf.layers.conv2d(inputs=max_pool1AR,filters=input_channels*16,
                             kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
    max_pool2AR = tf.layers.max_pooling2d(inputs=conv2AR, pool_size=[2, 2], strides=2)
    
    
    #Hourglass part -- DeConvolutional Layer1
    deConvFilter1 = tf.Variable(tf.truncated_normal([3, 3, input_channels*4, input_channels*16],stddev=0.1,dtype=tf.float32))
    deconv1 = tf.nn.conv2d_transpose(max_pool2AR,filter = deConvFilter1,
                              output_shape =tf.stack([tf.shape(x)[0], int(input_height/2), int(input_width/2),input_channels*4]),
                              strides = [1, 2, 2, 1], padding = "SAME")
    deconv1=tf.nn.relu(deconv1)
    
    ## Hourglass part -- DeConvolutional Layer2
    deConvFilter2 = tf.Variable(tf.truncated_normal([3, 3, input_channels, input_channels*4],stddev=0.1,dtype=tf.float32))
    deconv2 = tf.nn.conv2d_transpose(deconv1,filter = deConvFilter2,
                              output_shape =tf.stack([tf.shape(x)[0], int(input_height), int(input_width),input_channels]),
                              strides = [1, 2, 2, 1], padding = "SAME")
    deconv2=tf.nn.relu(deconv2)
    
    
    ## Then define the Trunk part - ResNet Block
    
    conv1Trunk=tf.layers.conv2d(inputs=x,filters=input_channels*4,
                             kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
    
    conv2Trunk=tf.layers.conv2d(inputs=conv1Trunk,filters=input_channels,
                             kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
    
    maskWeight=conv2Trunk+x
    
    ## Multplication of Masking and Trunk:
    MT= maskWeight*deconv2
    
    ## The addition to be considered as well
    final=MT+maskWeight
    
    return MT

x_input_shape = (None, 28, 28,1)
x_input = tf.placeholder(tf.float32, shape=x_input_shape)
y_target = tf.placeholder(tf.int32, shape=(None,))

# Declare model parameters 4x4 convolutional kernel sizes
conv1Dir=tf.layers.conv2d(inputs=x_input,filters=16,
                          kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
attenRes1=residualAttentionBlock(conv1Dir)

## Second Convolutional Layer -- Add a ResNet Block
conv2Dir=tf.layers.conv2d(inputs=attenRes1,filters=32,
                          kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
max_pool2Dir = tf.layers.max_pooling2d(inputs=conv2Dir, pool_size=[2, 2], strides=2)


conv2_2Dir=tf.layers.conv2d(inputs=max_pool2Dir,filters=32,
                          kernel_size=[3, 3],padding="same",activation=tf.nn.relu)

conv2ResBlk=conv2_2Dir+max_pool2Dir

conv3_1Dir=tf.layers.conv2d(inputs=conv2ResBlk,filters=64,
                          kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
#attenRes2=residualAttentionBlock(max_pool2_2Dir)

conv3Dir=tf.layers.conv2d(inputs=conv3_1Dir,filters=128,
                          kernel_size=[3, 3],padding="same",activation=tf.nn.relu)

max_pool3Dir = tf.layers.max_pooling2d(inputs=conv3Dir, pool_size=[2, 2], strides=2)
conv4Dir=tf.layers.conv2d(inputs=max_pool3Dir,filters=256,
                          kernel_size=[3, 3],padding="same",activation=tf.nn.relu)

conv5Dir=tf.layers.conv2d(inputs=conv4Dir,filters=512,
                          kernel_size=[3, 3],padding="same",activation=tf.nn.relu)

## flatten
flatpool1=tf.reshape(conv5Dir,[-1,7*7*512])
flatpool2=tf.layers.dense(flatpool1,1024,activation=tf.nn.sigmoid)
flatpool3=tf.layers.dense(flatpool2,128,activation=tf.nn.sigmoid)
encoded = tf.layers.dense(flatpool3, 10,activation=tf.nn.sigmoid)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=encoded, labels=y_target))
my_optimizer = tf.train.MomentumOptimizer(0.005, 0.9)
#my_optimizer = tf.train.AdamOptimizer(0.005)

train_step = my_optimizer.minimize(loss)

# Initialize Variables
init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)


# In[ ]:


## Make training batches
batchSize=100
numberOfEpoches=400
tempLoss=0.0
cnter=1
for i in range(numberOfEpoches):
    for batchIdx in range(int(train_xdata.shape[0]/100)):
        tempImg=train_xdata[batchIdx*100:(batchIdx+1)*100]
        tempLabel=train_labels[batchIdx*100:(batchIdx+1)*100]
        tploss,_=sess.run([loss,train_step],feed_dict={x_input: tempImg,y_target: tempLabel})
        tempLoss+=tploss
        cnter+=1
        if cnter%1000==0:
            print('Current loss: ',str(tploss/1000.0))
            tempLoss=0


# In[ ]:


tempImg=test_xdata
pred=sess.run([encoded],feed_dict={x_input: tempImg})


# In[ ]:


predRes=[np.argmax(x,axis=0) for x in pred[0]]


# In[ ]:


toSubmit=pd.DataFrame()
toSubmit['ImageId']=list(range(1,28001))
toSubmit['Label']=predRes


# In[ ]:


toSubmit.to_csv('toSubmit.csv',index=False)


# In[ ]:




