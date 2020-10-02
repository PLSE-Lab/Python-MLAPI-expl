#!/usr/bin/env python
# coding: utf-8

# **Summary**

# In this kernel I bulid CNN network to classify radigraph chest images into two classes: patients with pneumonia or withouth the sign of the disiease. To build CNN I used raw Tensorflow howevere to simplify some operation I will use Keras approach. I recieved 99% accuracy on my training set and 87% on validataion set. Testing model on test set gave me 77% accuracy.  
# 
# Main goal of this kernal was not accouracy of the model but approach to image processing and CNN construction in raw Tesnorflow. I describe operation on images like image augumentation or rescaling to get more training examples and make CNN work fast. 
# 
# This is the first part of my kernel. The second will describe very similar apporach in Keras library only where I will also implementy ready to use CNN like VGG16.

# **1. Data preprocesing**

# In this section I will make some data preporcesing. Mainly data augumentation to generate more samples, rescaling and reshaping np arrays to fit my CNN network. 

# Importing libraries 

# In[ ]:


import numpy as np
import cv2
import random
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt


# Constructing function which read dataset into X and y array. Arguments are type of data set (train, test,val), image width and image height. We create arrays where we upload our images in gray scale. Next we reshaping our arrays to appropriate size to fit our CNN. At the end we shuffeling our data set. 

# In[ ]:


def generate_set(set_type, img_size_width, img_size_height):
    X=[]
    y=[]
    
    path_file_pneumonia = os.path.join("../input/chest_xray/chest_xray",set_type,"PNEUMONIA/*.jpeg")
    path_pneumonia = glob.glob(path_file_pneumonia)

    for i in tqdm(path_pneumonia):
        file = cv2.imread(i,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(file,dsize=(img_size_width,img_size_height))
        X.append(img)
        y.append(1)
        

    path_file_normal = os.path.join("../input/chest_xray/chest_xray",set_type,"NORMAL/*.jpeg")
    path_normal = glob.glob(path_file_normal)

    for j in tqdm(path_normal):
        file_2 = cv2.imread(j,cv2.IMREAD_GRAYSCALE)
        img_2 = cv2.resize(file_2,dsize=(img_size_width,img_size_height))
        X.append(img_2)
        y.append(0)
      

    X = np.array(X, dtype ='float32').reshape(-1,img_size_width*img_size_height)
    y = np.array(y).reshape(-1,)
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]
    
    return X,y


# Lets read our data sets. 

# In[ ]:


X_train, y_train = generate_set("train", 150,150)
X_val,y_val = generate_set("val", 150,150)
X_test,y_test = generate_set("test", 150,150)


# Lets have a look on data sets shape

# In[ ]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(X_val.shape)
print(y_val.shape)


# If we take into consideration number of examples to learn our CNN it is not enough images to make satisfying predictions. We will perform some data augmentation for this classes where we have less examples than other class. In our case it is Normal class in train set.  

# In[ ]:


str(print("Pneumonia train: ")) + str(print(np.count_nonzero(y_train == 1)))

str(print("Normal train: ")) + str(print(np.count_nonzero(y_train == 0)))


# Using 2 generators from keras we will add to train set 2534 new generated examples for the class normal. In this case I decided to use Keras due to easy to use interface. In my next kernel I will try to build the same CNN using only Keras library to compare simplicity between Keras and Tensorflow libs. Full explanation of the function below you can find here: https://keras.io/preprocessing/image/

# Below I build 2 diffrent generators. Each one will generate 1267 new examples separetly. Here is a short explanantion about parameters used:
# * rotation_range give us angle range to rotate our image randomly. 
# * width_shift_range and height_shift_range set the ranges in the image area where random transfomration will be made.
# * shear_range - image cropped random range. 
# * zoom_range: ranodm range for zooming part of images. 
# * horizontal_flip - a random reflection of half of the image in a horizontal plane.
# * fill_mode - strategy to fill new pixels

# In[ ]:


from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

datagen_1 = ImageDataGenerator( 
    rotation_range=40,
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True, 
    fill_mode='nearest',
    rescale= 1./255)

datagen_2 = ImageDataGenerator( 
    rotation_range=20,
    width_shift_range=0.1, 
    height_shift_range=0.3, 
    shear_range=0.3, 
    zoom_range=0.3, 
    horizontal_flip=True, 
    fill_mode='nearest',
    rescale=1./255)


# Lets take only normal class images from our train set. 

# In[ ]:


idx = np.where(y_train == 0)
X_train_normal= X_train[idx]


# Reshape images and implement generator to our data. 

# In[ ]:



X_train_normal = X_train_normal.reshape(len(X_train_normal),150,150,1)

itr = train_generator = datagen_1.flow(X_train_normal, batch_size = 1267)
X_1 = itr.next()
X_1 = np.array(X_1, dtype ='float32').reshape(-1,150*150)

itr = train_generator = datagen_2.flow(X_train_normal, batch_size = 1267)
X_2 = itr.next()
X_2 = np.array(X_2, dtype ='float32').reshape(-1,150*150)

X = np.concatenate((X_1,X_2), axis=0)
y_tmp = []
y_tmp = [0]*(1267*2)


# Our numpy array as we see below was transformed correctly. Image was rescaled in to the range 0-1. 

# In[ ]:


X[0].reshape(150,150)


# Lets only rescale to 0-1 range our X_train. We simply devide every pixel value by 255. 

# In[ ]:


X_train = X_train/255


# In[ ]:


X_train[0]


# Lets show oryginal and generated images. 

# In[ ]:


X_train[0].reshape(150,150)
imgplot = plt.imshow(X_train[0].reshape(150,150))
plt.title("Oryginal")
plt.show()


X[0].reshape(150,150)
imgplot = plt.imshow(X[0].reshape(150,150))
plt.title("Generated")
plt.show()


# Lets concatenate our numpy arrays and shuffle obseravations. 

# In[ ]:


X_train_ag = np.concatenate((X_train,X), axis=0)
y_train_ag = np.concatenate((y_train,y_tmp), axis=0)
idx = np.random.permutation(len(X_train_ag))
X_train_ag = X_train_ag[idx]
y_train_ag = y_train_ag[idx]
X_train_ag.shape


# In[ ]:


str(print("Pneumonia train: ")) + str(print(np.count_nonzero(y_train_ag == 1)))

str(print("Normal train: ")) + str(print(np.count_nonzero(y_train_ag == 0)))


# Lets also transform our test and val sets. Only rescaling to 0-1.  

# In[ ]:


X_test = X_test/255
X_val = X_val/255


# In[ ]:


X_val[0]


# **2. Bulding CNN**

# The parameters of my network are as fallow: 
# * Adam algorythm to optimize my loss function 
# * Softmax cross entropy with logits as my cost function 
# * input shape of images 150x150x1 
# * First conv layer: 32 filters, kernel window 3x3, relu activation function, padding without filling (valid)
# * Maxpooling: reduce 2 times, with step (stride) 2 
# * Second conv layer: 64 filters, kernel window 3x3, relu activation function, padding without filling (valid)
# * Maxpooling: reduce 2 times, with step (stride) 2 
# * Third conv layer: 128 filters, kernel window 3x3, relu activation function, padding without filling (valid)
# * Maxpooling: reduce 2 times, with step (stride) 2 
# * Fourth conv layer: 128 filters, kernel window 3x3, relu activation function, padding without filling (valid)
# * Maxpooling: reduce 2 times, with step (stride) 2 
# * Flatting layer 
# * Dense layer 512 neurons with relu activation
# * Dropout with 0.5 
# * logits with 2 neurons and softmax as a activation 

# In[ ]:


import numpy as np
import os

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# In[ ]:




import tensorflow as tf

height = 150
width = 150
channels = 1
n_inputs = height * width * channels

reset_graph()

X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
y = tf.placeholder(tf.int64, shape=[None], name="y")
training = tf.placeholder_with_default(False, shape=[], name='learning')

conv1 = tf.layers.conv2d(X_reshaped, filters=32, kernel_size=[3,3],
                         padding="VALID",
                         activation=tf.nn.relu, name="cnn_1")
#conv1 size(148,148,32)

pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2,2], strides=2, name="maxpool1")
#pool1 size(74,74,32)

conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=[3,3],
                        padding="VALID",
                        activation=tf.nn.relu, name="cnn_2")
#conv2 size(72,72,64)

pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2,2], strides=2, name="maxpool2")
#pool2 size(36,36,64)

conv3 = tf.layers.conv2d(pool2, filters=128, kernel_size=[3,3],
                        padding="VALID",
                        activation=tf.nn.relu, name="cnn_3")

#conv3 size(34,34,128)

pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2,2], strides=2, name="maxpool3")

#pool3 size(17,17,128)

conv4 = tf.layers.conv2d(pool3, filters=128, kernel_size=[3,3],
                        padding="VALID",
                        activation=tf.nn.relu, name="cnn_4")
#conv4 size(15,15,128)
pool4 = tf.layers.max_pooling2d(conv4, pool_size=[2,2], strides=2, name="maxpool4")

#conv4 size(7,7,128)

pool_flat = tf.reshape(pool4, [-1, 7 * 7 * 128])
#pool_flat size 6272

dense1 = tf.layers.dense(inputs=pool_flat, units=512, activation=tf.nn.relu)
#dense1 512 

fc1_drop = tf.layers.dropout(dense1, 0.5, training=training)

logits = tf.layers.dense(fc1_drop, 2, name="output")
Y_proba = tf.nn.softmax(logits, name="Y_probability")

with tf.name_scope("learning"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("acc"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("saver"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


# **3.Training CNN**

# Below I defined batch function to generate random samples to train my CNN and validate outcomes. 

# In[ ]:


def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


# I decided to set batch_size to 500 and number of epochs to 20.

# In[ ]:



n_epochs = 20
batch_size = 500

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(len(X_train) // batch_size):
            X_batch, y_batch = next_batch(batch_size,X_train,y_train)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_val, y: y_val})
        print(epoch, "Acc train set:", acc_train, "Acc val set:", acc_test)

        save_path = saver.save(sess, "./model_tensorflow.ckpt")


# Next lets read our model again and predict on test set. 

# In[ ]:


with tf.Session() as sess:
    saver.restore(sess, "./model_tensorflow.ckpt")
    X_new_scaled = X_test
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print("Acc on test set: ", acc_test)

