#!/usr/bin/env python
# coding: utf-8

# # In this notebook i tried to cover basic implementation of Tensorflow with Hidden layers. So I hope you guys like it. Upvote this kernel.  
# ## In this notebook you will learn following:
# 1.  Basic importing and data preprocessing for Deep learning
# 2.  Creating Variables such as variable for weight (w) and bias (b)
# 3.  Hidden layers creation
# 4.  Implementation of mini batch for training 
# 5. Testing accuracy

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# ## 1. Importing files 

# In[ ]:


# Importing CSV files as this step you all knows  
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# In[ ]:


# Checking dataframe shape
train.shape , test.shape


# In[ ]:


train.head()


# ### Since 1st column of dataframe train gives label detail. So lets extract that label column

# In[ ]:


# Extracting label from train dataframe
train_label=train.iloc[:,0]


# ### For further use we need to convert our train_label dataframe to Hot - Encode 

# In[ ]:


#converting dataframe to category for hot encode
train_label=train_label.astype('category')


# In[ ]:


# Converted into Hot encode
train_label=pd.get_dummies(train_label)
train_label.shape


# In[ ]:


del train['label']


# ## Now Lets start import tensorflow

# In[ ]:


# importing Tensorflow
import tensorflow as tf


# ## Now create variable with initializer along with their shape and make your logit with equation
# ## (input * weight)+bias
# 

# In[ ]:


def variable(x,weight_shape,bias_shape):
    weight_init=tf.truncated_normal_initializer(stddev=0.1)
    bias_init=tf.constant_initializer(0.1)
    weight=tf.get_variable(shape=weight_shape,name='weight',initializer=weight_init)
    bias=tf.get_variable(shape=bias_shape,name='bias',initializer=bias_init)
    output= tf.add(tf.matmul(x,weight),bias)
    return output


# ## Now placeholder which is use to initialize once when graph is run. Basically placeholder is use for giving input to NeuralNet

# In[ ]:


x=tf.placeholder(tf.float32,name='x',shape=[None,784])
y=tf.placeholder(tf.float32,name='y',shape=[None,10])
drop=tf.placeholder(tf.float32)


# ## We are going to make a Neural Network with 3 Hidden layer . 1st Hidden layer consist of 512 neurons then 2nd hidden layer of 256 neurons then last hidden layer of 128 neurons which ultimately gives final output with 10 softmax layer neurons

# ![](https://assets.digitalocean.com/articles/handwriting_tensorflow_python3/cnwitLM.png)

# In[ ]:


with tf.variable_scope('layer_1'):
    hidden_1=variable(x,[784,512],[512])
with tf.variable_scope('layer_2'):
    hidden_2=variable(hidden_1,[512,256],[256])
with tf.variable_scope('layer_3'):
    hidden_3=variable(hidden_2,[256,128],[128])
    out1=tf.nn.dropout(hidden_3,drop)   # To prevent from Overfitting
with tf.variable_scope('outputlayer'):
    output=variable(out1,[128,10],[10])


# ## Now defining Loss function for gradientDescent optimizer for getting optimal value of logit variables

# In[ ]:


# Defining cost function which will be used  by gradient descent
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=output))   


# ## Using AdamOptimizer to minimize cross entropy. You can also use GradientDescentOptimizer inplace of AdamOptimizer

# In[ ]:


# Graident Descent for minimize cross entropy
optimize=tf.train.AdamOptimizer(learning_rate=0.001)
step=optimize.minimize(cross_entropy)


# 
# ## Checking accuracy with foloowing

# In[ ]:


correct_pred=tf.equal(tf.argmax(output,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))


# In[ ]:


# Initialization of all variables
init=tf.initialize_all_variables()


# In[ ]:


train.shape, test.shape


# In[ ]:


# Creating Session for all computation 
sess=tf.Session()


# In[ ]:


#Initialize variables
sess.run(init)


# ## Converting Dataframe into ndarray

# In[ ]:


train=train.values
train_label=train_label.values


# In[ ]:


# Some useful parameters for minibatch creation and gradient descent optimization 
iteration=2000
batch_size=256


# ## Below we are going to divide data into Mini batches and Iterate over many times to get global minimum cost

# In[ ]:


for i in range(iteration):
    choice=np.random.choice(42000,size=batch_size)
    x_train=train[choice]
    y_train=train_label[choice]
    sess.run(step,feed_dict={x:x_train,y:y_train,drop:0.4})

    if (i%100==0):
        loss,accu=sess.run([cross_entropy,accuracy],feed_dict={x:x_train,y:y_train,drop:1})
        print ('loss is',str(loss),'accuracy is',str(accu),'iteration',i+1)


# # Now after training our model let's check it's accuracy

# In[ ]:


# Test dataset shape
test.shape


# In[ ]:


# Convert to arrays
test_array=test.values


# ## Now start session for prediction

# In[ ]:


result=sess.run(output,feed_dict={x:test_array,drop:1})


# In[ ]:


result=np.argmax(result,axis=1)


# In[ ]:


final=pd.DataFrame({'Predicted':result})
final.head()


# ## Lets check for first 3 prediction made by model

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


arr=test_array[0:5]
i=0
plt.imshow(arr[i].reshape([28,28]))
plt.title(final.iloc[0,0],size=20)


# In[ ]:


arr=test_array[0:5]
i=1
plt.imshow(arr[i].reshape([28,28]))
plt.title(final.iloc[1,0],size=20)


# In[ ]:


arr=test_array[0:5]
i=2
plt.imshow(arr[i].reshape([28,28]))
plt.title(final.iloc[2,0],size=20)


# # So our model performed well. 

# # Thanks. :) Happy learning

# 
