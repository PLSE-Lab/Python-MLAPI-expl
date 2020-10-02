#!/usr/bin/env python
# coding: utf-8

# # LeNet-5 Architecture 

# Convolutional Neural Networks are are a special kind of multi-layer neural networks. Like almost every other neural networks they are trained with a version of the back-propagation algorithm. Where they differ is in the architecture.   
# Convolutional Neural Networks are designed to recognize visual patterns directly from pixel images with minimal preprocessing.  
# They can recognize patterns with extreme variability (such as handwritten characters), and with robustness to distortions and simple geometric transformations.  
# 
# 
# ##### <span style="color:orange">LeNet-5</span>  is a  convolutional network designed by  <span style="color:orange">Yann LeCun</span>  for handwritten and machine-printed character recognition.  

# ![Imgur](https://i.imgur.com/VDOzp4t.jpg)

# ## Importing Libraries

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# ## Read dataset

# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


train.head()


# In[ ]:


train.shape


# ## Preprocessing

# In[ ]:


data_X = train.drop('label', axis=1).values
data_Y = train['label'].values


# In[ ]:


data_Y = np.eye(10)[data_Y]     #one hot encoding

data_X = data_X/255.0             #normalization
data_X = data_X.reshape(-1, 28, 28, 1)


# The images in the dataset have been given in 28x28 resolution. Since Lenet - 5 Architecture requires 32x32 images, we would convert the given images into 32x32. In order to do so, we would apply extra zero padding on the images.

# In[ ]:


data_X = np.pad(data_X, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')     # pad


# In[ ]:


print('Shape of data_X : ', data_X.shape)
print('Shape of data_Y : ', data_Y.shape)


# In[ ]:


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_X, data_Y, test_size = 0.33)


# In[ ]:


print('Shape of train_x \t: ', train_x.shape)
print('Shape of train_y \t: ', train_y.shape)
print('Shape of test_x  \t: ', test_x.shape)
print('Shape of test_y  \t: ', test_y.shape)


# ## Helper Functions

# In[ ]:


# Outputs a batch of data
def next_batch(batch_size, x, y):
    
    idx = np.arange(0, len(x))
    np.random.shuffle(idx)
    
    idx = idx[: batch_size]
    
    x = [x[i] for i in idx]
    y = [y[i] for i in idx]
    
    x = np.asarray(x)
    y = np.asarray(y)
    return x, y


# In[ ]:


# Display 10 random images
def display_images(data, title, display_label = True):
    x, y = data
    x = x.reshape(-1, 32, 32)
    y = np.argmax(y, axis=1)
    fig, axes = plt.subplots(2, 5, figsize=(15, 5))
    fig.subplots_adjust(hspace = 0.7, wspace = 0.7)
    fig.suptitle(title, fontsize = 18)
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i], cmap='gray')
        if display_label:
            ax.set_xlabel(y[i], fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


# In the original paper tanh activation function was used but in modern times relu activation function provides better results.

# In[ ]:


# wrapper function for convolution layer
def conv2d(x, w, b, strides = 1, padding = 'VALID'):
    x = tf.nn.conv2d(x, w, strides = [1, strides, strides, 1], padding = padding)
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)
    return x


# In[ ]:


# wrapper function for pooling layer
def maxpool2d(x, k = 2, padding = 'VALID'):
    x = tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, k, k, 1], padding = padding)
    return x 


# In[ ]:


# wrapper function for fully connected layer
def dense(x, w, b, activation = 'none'):
    x = tf.matmul(x, w)
    x = tf.nn.bias_add(x, b)
    if activation == 'relu':
        x = tf.nn.relu(x)  
    return x


# ## Visualization

# Display 10 random images and their labels from the training data

# In[ ]:


display_images(next_batch(10, train_x, train_y), 'Training Images')


# ## Model

# In[ ]:


# training hyperparameters
learning_rate = 0.001
epochs = 10000
batch_size = 128


# In[ ]:


# image dimensions and number of classes
image_width = 32
image_height = 32
color_channels = 1
n_classes = 10


# In[ ]:


# placeholders
X = tf.placeholder(tf.float32, shape = [None, image_width, image_height, color_channels]) 
Y = tf.placeholder(tf.float32, shape = [None, n_classes]) 


# In[ ]:


# parameters
weights = { 
    'w1' : tf.Variable(tf.random_normal([5, 5, 1, 6])), 
    'w2' : tf.Variable(tf.random_normal([5, 5, 6, 16])), 
    'w3' : tf.Variable(tf.random_normal([400, 120])),
    'w4' : tf.Variable(tf.random_normal([120, 84])),
    'w5' : tf.Variable(tf.random_normal([84, 10])) 
}

biases = {
    'b1' : tf.Variable(tf.random_normal([6])),
    'b2' : tf.Variable(tf.random_normal([16])),
    'b3' : tf.Variable(tf.random_normal([120])),
    'b4' : tf.Variable(tf.random_normal([84])),
    'b5' : tf.Variable(tf.random_normal([10]))
}


# In[ ]:


# the model
def lenet(x, weight, bias):
    
    # Layer 1 : Convolution : input (32x32x1); output (28x28x6)
    conv1 = conv2d(x, weight['w1'], bias['b1'])
    # Layer 2 : Sub sampling : input (28x28x6); outpute (14x14x6)
    pool1 = maxpool2d(conv1) 
    # Layer 3 : Convolution : input (14x14x6); output (10x10x16)
    conv2 = conv2d(pool1, weight['w2'], bias['b2'])
    # Layer 4: Sub sampling : input (10x10x16); output (5x5x16)
    pool2 = maxpool2d(conv2) 
    # Flatten the 2-d image to a 1-d vector : input (5x5x16); output (400)
    flat = tf.reshape(pool2, [-1, weights['w3'].get_shape().as_list()[0]])
    # Layer 5: Fully Connected : input (400); output (120)
    fc1 = dense(flat, weight['w3'], bias['b3'], 'relu')
    # Layer 6: Fully Connected : input (120); output (84)
    fc2 = dense(fc1, weight['w4'], bias['b4'], 'relu')
    # Layer 7: Fully Connected : input (84); output (10)
    logits = dense(fc2, weight['w5'], bias['b5'], 'none') 

    
    return logits


# In[ ]:


# model output
logits = lenet(X, weights, biases)
# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
# minimize loss
train = optimizer.minimize(loss)


# In[ ]:


correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[ ]:


init = tf.global_variables_initializer()


# ## Train

# In[ ]:


with tf.Session() as sess:
    
    sess.run(init)
    
    cost_hist, acc_hist = [], []
    
    for epoch in range(1, epochs + 1):
        
        batch_x, batch_y = next_batch(batch_size, train_x, train_y)
        
        sess.run(train, feed_dict = { X : batch_x, Y : batch_y})
    
        if epoch % 500 == 0:
            c, acc = sess.run([loss, accuracy], feed_dict = { X : batch_x, Y : batch_y})
            cost_hist.append(c)
            acc_hist.append(acc)
            print('Epoch ' + str(epoch) + ', Cost: ' + str(c) + ', Accuracy: ' + str(acc))

    print('-' * 70)
    print('\nOptimization Finished\n')
    print('Accuracy on train data \t: ' + str(sess.run(accuracy, feed_dict = { X : train_x, Y :train_y}) * 100) + ' %')
    print('Accuracy on test data  \t: ' + str(sess.run(accuracy, feed_dict = { X : test_x, Y : test_y}) * 100) + ' %')


# ### Change in cost function

# In[ ]:


eps = list(range(500, epoch+500, 500))
plt.plot(eps, cost_hist)
plt.title("Change in cost")
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.show()


# ### Change in accuracy

# In[ ]:


eps = list(range(500, epoch+500, 500))
plt.plot(eps, acc_hist)
plt.title("Change in accuracy")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()


# ## Validate

# In[ ]:


# KFold cross validation
from sklearn.model_selection import KFold

kf = KFold(n_splits = 5)

current_fold = 1

train_acc_hist = []
test_acc_hist = []

for train_index, test_index in kf.split(data_Y):
     
    KFold_X_train = data_X[list(train_index)]
    KFold_X_test = data_X[test_index]
    KFold_Y_train = data_Y[train_index]
    KFold_Y_test = data_Y[test_index]

    # run the graph
    with tf.Session() as sess:
    
        sess.run(init)

        for epoch in range(1, epochs + 1):

            batch_x, batch_y = next_batch(batch_size, KFold_X_train, KFold_Y_train)

            sess.run(train, feed_dict = { X : batch_x, Y : batch_y })
    
        train_accuracy = sess.run(accuracy, feed_dict = { X : KFold_X_train, Y :KFold_Y_train }) * 100
        test_accuracy = sess.run(accuracy, feed_dict = { X : KFold_X_test, Y : KFold_Y_test }) * 100
        
        train_acc_hist.append(train_accuracy)
        test_acc_hist.append(test_accuracy)

        print('\nFOLD-' + str(current_fold) + '\n')
        print('Accuracy on train data \t:  {0:.2f} %'.format(train_accuracy))
        print('Accuracy on test data  \t:  {0:.2f} %'.format(test_accuracy))
        
        
        W = sess.run(weights)
        B = sess.run(biases)
    

    current_fold = current_fold +1
    
train_cross_val_score = np.mean(train_acc_hist)    
test_cross_val_score = np.mean(test_acc_hist)

print('-' * 70)
print('\nOPTIMIZATION FINISHED\n')
print('\n\nFINAL TRAIN SET K-FOLD CROSS VALIDATION ACCURACY \t:  {0:.2f}'.format(train_cross_val_score))
print('\nFINAL TEST SET K-FOLD CROSS VALIDATION ACCURACY    \t:  {0:.2f}'.format(test_cross_val_score))


# ## Save the parameters of the model

# In[ ]:


for key in W.keys():
    np.save(key, W[key])
for key in B.keys():
    np.save(key, B[key])


# ## Predict

# In[ ]:


# load the prediction dataset
test = pd.read_csv('../input/test.csv')

# preprocess the prediction dataset 
data_X = test.values
data_X = data_X.reshape(-1, 28, 28, 1)     # reshape
data_X = np.pad(data_X, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')     # pad
data_X = data_X/255.0     # normalize


# In[ ]:


# set the parameters to the learned parameters
for key in weights.keys():
    weights[key] = tf.Variable(W[key])
for key in biases.keys():
    biases[key] = tf.Variable(B[key])


# In[ ]:


logits = lenet(X, weights, biases)


# In[ ]:


with tf.Session() as sess:
    
    for key in weights.keys():
        sess.run(weights[key].initializer)
    for key in biases.keys():
        sess.run(biases[key].initializer)
    
    output = sess.run(logits, feed_dict= {X : data_X}) 


# ## Visualization

# Display the images and their labels as predicted by the LeNet-5 model

# In[ ]:


display_images(next_batch(10, data_X, output), 'Prediction Images')


# In[ ]:


predictions = np.argmax(output, axis=1)


# ## Kaggle Submission

# Code to make the submission file for kaggle

# In[ ]:


index = test.index +1 

ImageId = list(index)
Label = list(predictions)

df = pd.DataFrame([ImageId ,Label])
df = df.T
df.columns = [['ImageId', 'Label']]

df.to_csv('submission.csv', index=False)


# In[ ]:




