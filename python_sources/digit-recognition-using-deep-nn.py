#!/usr/bin/env python
# coding: utf-8

# Please if you like this kernel give some encouragement by an upvote.

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sbn

import tensorflow as tf

from sklearn.model_selection import train_test_split


# # Digit recognition using Neural Networks

# In[ ]:


train_data = pd.read_csv('../input/train.csv',)
test_data = pd.read_csv('../input/test.csv')


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


label_df = train_data['label']
train_data.drop(['label'],inplace=True,axis=1)
train_labels = pd.get_dummies(label_df,columns= ['label'])

train_labels[train_labels.columns] = train_labels[train_labels.columns].                                        astype(np.float)


# In[ ]:


print(f"shape of train_labels = {train_labels.shape}")
print(f"shape of train_data = {train_data.shape}")
print(f"shape of test_data = {test_data.shape}")


# set the intial hyper parameters

# In[ ]:


#images = train_data.iloc[:,1:].values
images = train_data.astype(np.float)

images = np.multiply(images,1.0/255)
test_images = np.multiply(test_data, 1.0 / 255.0)
test_images = test_images.astype(np.float)
print(f"dimensions of the images dataframe {images.shape}")
print(f"dimensions of the images dataframe {test_images.shape}")


# In[ ]:


image_height = image_width = np.ceil(np.sqrt(images.shape[1])).astype(np.uint8)


# In[ ]:


print(f"height of the image {image_height}")
print(f"width of each image {image_width}")


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(images,
                                                 train_labels,
                                                 test_size = 0.20, 
                                                 random_state = 42
                                                )
x_train = x_train.reset_index(drop=True)
x_test = x_test.reset_index(drop = True)
y_test = y_test.reset_index(drop = True)
y_train = y_train.reset_index(drop = True)



# In[ ]:


print(f"shape of train_x = {x_train.shape}, y_train = {y_train.shape}")
print(f"shape of test_x = {x_test.shape}, y_test= {y_test.shape}")


# In[ ]:


x_train.head()


# In[ ]:


epochs_completed = 0
index_in_epoch = 0
num_examples = x_train.shape[0]

def next_batch(batch_size):
    
    global x_train
    global y_train
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        x_train = x_train.iloc[perm]
        y_train = y_train.iloc[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return x_train[start:end], y_train[start:end]


# In[ ]:


x = tf.placeholder('float',shape=[None,784])
y_ = tf.placeholder('float',shape=[None,10])
batch_size = tf.placeholder('float')
keep_prob = tf.placeholder('float')

w = tf.Variable(tf.zeros([784,10]),name='Weights')
b = tf.Variable(tf.zeros([10]),name= 'bias')

y = tf.nn.softmax(tf.matmul(x,w)+b)

cros_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_,
                                                      logits = y))


# **Test helper functions**

# In[ ]:


train_step = tf.train.            GradientDescentOptimizer(0.5).            minimize(cros_entropy)    

correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)


# In[ ]:


# # visualisation variables
# train_accuracies = []
# validation_accuracies = []
# x_range = []
# TRAINING_ITERATIONS = 1000
# DROPOUT = 0.5

# display_step = 1
# for i in range(TRAINING_ITERATIONS):

#     #get new batch
#     batch_xs, batch_ys = next_batch(BATCH_SIZE)        

    
#     if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:
        
#         train_accuracy = accuracy.eval(feed_dict={x:batch_xs, 
#                                                   y_: batch_ys, 
#                                                   keep_prob: 1.0})       
#         if(i%100 == 0):
#             validation_accuracy = accuracy.eval(feed_dict={ x: x_test[0:BATCH_SIZE], 
#                                                             y_: y_test[0:BATCH_SIZE], 
#                                                             keep_prob: 1.0})                                  
#             print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))
            
#             validation_accuracies.append(validation_accuracy)
            
#         else:
#              print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
#         train_accuracies.append(train_accuracy)
#         x_range.append(i)
        
#         # increase display_step
#         if i%(display_step*10) == 0 and i:
#             display_step *= 10
#     # train on batch
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})


# Lets start with Deeplearning

# In[ ]:


x_image = tf.reshape(x,[-1,28,28,1],name='x_image')


# In[ ]:


def weight_variable(dim):
    w_init = tf.truncated_normal(dim,stddev=0.1)
    return tf.Variable(w_init)

def bias_variable(dim):
    b_init = tf.constant(0.2, shape = dim,)
    
    return tf.Variable(b_init)


# In[ ]:


def conv_mnist(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_mnist(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],
                         strides= [1,2,2,1], padding = 'SAME')


# In[ ]:


keep_prob = tf.placeholder('float')
BATCH_SIZE = 50
LEARNING_RATE = 1e-3

W_con1 = weight_variable([5,5,1,32])
b_con1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv_mnist(x_image,W_con1)+b_con1)
h_pool1= max_pool_mnist(h_conv1)

W_con2 = weight_variable([5,5,32,64])
b_con2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv_mnist(h_pool1,W_con2)+b_con2)
h_pool2= max_pool_mnist(h_conv2)


W_con3 = weight_variable([7*7*64,1024])
b_con3 = bias_variable([1024])

h_pool_flat1 = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat1,W_con3)+b_con3)

h_fc_drop = tf.nn.dropout(h_fc1,keep_prob)

W_con_fin = weight_variable([1024,10])
b_con_fin = bias_variable([10])


y_conv = tf.matmul(h_fc_drop,W_con_fin) + b_con_fin

cros_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_,
                                                      logits = y_conv))

train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cros_entropy)    

correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y_conv,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)


# In[ ]:


# visualisation variables
train_accuracies = []
validation_accuracies = []
x_range = []
TRAINING_ITERATIONS = 3000
DROPOUT = 0.2

display_step = 1
for i in range(TRAINING_ITERATIONS):

    #get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)        

    
    if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:
        
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs, 
                                                  y_: batch_ys, 
                                                  keep_prob: 1.0})       
        if(i%100 == 0):
            validation_accuracy = accuracy.eval(feed_dict={ x: x_test[0:BATCH_SIZE], 
                                                            y_: y_test[0:BATCH_SIZE], 
                                                            keep_prob: 1.0})                                  
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))
            
            validation_accuracies.append(validation_accuracy)
            
        else:
             print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
        train_accuracies.append(train_accuracy)
        x_range.append(i)
        
        # increase display_step
        if i%(display_step*10) == 0 and i:
            display_step *= 10
    # train on batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})


# In[ ]:


# using batches is more resource efficient
predict = tf.argmax(y_conv,1)
predicted_lables = np.zeros(test_images.shape[0])

for i in range(0,test_images.shape[0]//BATCH_SIZE):
    predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={x: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE], 
                                                                                keep_prob: 1.0})


print('predicted_lables({0})'.format(len(predicted_lables)))


# save results
np.savetxt('results.csv', 
           np.c_[range(1,len(test_images)+1),predicted_lables], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')

