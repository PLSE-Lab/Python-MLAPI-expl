#!/usr/bin/env python
# coding: utf-8

# **Quick Draw!
# **
# This note book is just for tracking my own progress. Sharing with the though it might be helpful for other starters like me.  If you are looking for serious calculation please jump to the section with heading "Serious stuff"

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import ast
import cv2
import gc
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# **Attributes and their explanation**: Modified from https://github.com/googlecreativelab/quickdraw-dataset#the-raw-moderated-dataset 
# 1. key_id: Just a ID to identify each image. 
# 1. countrycode: two letter country code. More: https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2 
# 1. drawing: vectors of the doodle.
# 1. recognized: Whether the word was recognized by the game. If it is not recongnized by the game we can still use this labelled data for training the model.
# 1. time-stamp: Time 
# 1. word: the response class
# 
# [ 
#   [  // First stroke 
#     [x0, x1, x2, x3, ...],
#     [y0, y1, y2, y3, ...],
#     [t0, t1, t2, t3, ...]
#   ],
#   [  // Second stroke
#     [x0, x1, x2, x3, ...],
#     [y0, y1, y2, y3, ...],
#     [t0, t1, t2, t3, ...]
#   ],
#   ... // Additional strokes
# ]
# 
# Where x and y are the pixel coordinates, and t is the time in milliseconds since the first point. x and y are real-valued while t is an integer. The raw drawings can have vastly different bounding boxes and number of points due to the different devices used for display and input.
# 
# In simplified data time was removed. 

# In[ ]:


# Exploring train csv files to know the structure/dimensions of the data

data_train = pd.read_csv('../input/train_simplified/sleeping bag.csv',
                   index_col='key_id',
                   nrows=5)
data_train.head(5)


# **Test Data Set**

# In[ ]:


# Exploring test csv files to know the structure/dimensions of the data

data_test = pd.read_csv('../input/test_simplified.csv',
                   index_col='key_id',
                   nrows=5)
data_test.head(5)


# In[ ]:


# Looking at one Training data in details
# max value is 255. 
# import ast
words = data_train['word'].tolist()
drawings = [ast.literal_eval(pts) for pts in data_train[0:4]['drawing'].values] # python list


# In[ ]:


plt.figure(figsize=(10, 10))
for i, drawing in enumerate(drawings):
    plt.subplot(220 + (i+1))
    for x, y in drawing:
        plt.plot(x, y, marker='.')
        plt.title(words[i])


# **Creating Binary images from drawings:**
# Above images are still strokes, not images. So now we will convert these strokes to images. We will make images of size 255 by 255 for at least 100. One important resource is: Bresenham's Line Algorithm
#  http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm#Python, I modified the code from Miha Skalic for converting image. Thanks Miha.

# In[ ]:


# import cv2
def draw2img(drawing, Xsize=256, Ysize=256, lw = 4):
    '''
    converts drawing to image
    input:
    drawing: the drawing, it takes one at a time.
    Xsize, Ysize are the pixel size of the image.
    lw is the line width of the output image strokes.
    Note: This function uses cv2, so import cv2
    Example: >>>a = draw2img(drawings[3], 256, 256)
             >>>plt.imshow(a)
    '''
    fig, ax = plt.subplots()
    for x,y in drawing:
        ax.plot(x, y ,linewidth=lw) #  marker='.', See which line width is better, <4 might be good
    ax.axis('off')
    fig.canvas.draw()    
    X = np.array(fig.canvas.renderer._renderer)
    plt.close(fig)
    # image resizing. Original X is of various size due to strokes variable's length
    return (cv2.resize(X, (Xsize,Ysize)) / 255.)[::-1]


# In[ ]:


image = draw2img(drawings[3], 256, 256)
plt.imshow(image)


# In[ ]:


plt.figure(figsize=(10, 10))
for i in range(4):
    plt.subplot(140+i+1)
    plt.imshow(image[:, :, i])
    plt.title(i)


# From the above we see that color information is kept on the third dimension. Which is not necessary. So we will make a black and white one or at least a 2D one. Image is becoming a 3D one because we have used plotting function earlier. During plot function it created color images. Each stroke got a different color. to get rid of it. we can suppress it to black and white color.  Here in this constructed binary image '1' is bright spot and "0" is for dark spot. So our final b/w image we shoud be able to see only 1/0 in a 2D tensor. 

# In[ ]:


img = image[:,:,1] # taking only green channel
plt.imshow(img)
img.shape


# In[ ]:


# modifying the drawing to image function

def draw2img(drawing, shape =(32,32), lw = 4):
    '''
    converts drawing to image
    input:
    drawing: the drawing, it takes one at a time.
    Xsize, Ysize are the pixel size of the image.
    lw is the line width of the output image strokes.
    Note: This function uses cv2, so import cv2
    Example: >>>a = draw2img(drawings[3], 256, 256)
             >>>plt.imshow(a)
    '''
    fig, ax = plt.subplots()
    #drawing = eval(drawing)
    for x,y in drawing:
        ax.plot(x, y,'g',  marker='.', linewidth = lw) #  marker='.', See which line width is better, <4 might be good
    ax.axis('off')
    fig.canvas.draw()    
    X = np.array(fig.canvas.renderer._renderer)
    plt.close(fig)
    # image resizing. Original X is of various size due to strokes variable's length
    temp = (cv2.resize(X, shape) / 255.)[::-1]
    return temp[:,:,1] # only green channel, as we have drawn with green


# In[ ]:


img = draw2img(drawings[3], (256, 256), lw =1)
plt.imshow(img)
img.shape


# In[ ]:


'''
filename = list()
shape = list()
for file in os.listdir('../input/train_simplified/'):
    train = pd.read_csv('../input/train_simplified/' + file, index_col='key_id')
    filename.append(file.split('.')[0].replace(" ", "_"))
    shape.append(train.shape[0]) 
size_train = pd.DataFrame(data = shape, index =filename , columns  = ['entry'])

size_train.to_csv('training_sample_per_class.csv')
size_train.sort_values(by = 'entry')
'''


# Panda has 113613 entry and snow man has 340029 entry. Rest of the classes are with sample size in between these two numbers. 
# 

# **So far, and what next<br>**
# Now that we have a way to convert our strokes to images of any size we want, we will focus on how to use them for the next steps for image classification. Since all the files are training folder coming seperately for each class of image, we need to concatenate them to a master trainig data. Also before jmping into the serious business lets, make a toy training set of. In this toy we will not train on the whole trainng set, rather a subset of the trainig set. But we will train for all the classes. This toy will give us opportunity to simulate the whole thing without too much computations. It will be faster way of developing the whole system. In the final state we can train on whole data set with longer time. <br>
# Steps we will follow here, <br>
# * Cleaning
# * Making Trainin set
# * Build model using tensorflow convnet
# * Train the model
# * Evaluate the model
# 
# Few things we will keep in mind in this immediate next step. These are to save time and resource at this development stage.
# * drawing to image: we will use low resolution as 32*32 pixels jus to see.
# * Training sample size 1000 per class, just to play
# * We will take all 340 classes of images. 

# * Cleaning and preparing training data frame: 

# In[ ]:


# this is defined again because of reading training set again, where we will not use ast.literal_eval
def draw2img(drawing, shape =(32,32), lw = 1):
    '''
    converts drawing to image
    input:
    drawing: the drawing, it takes one at a time.
    Xsize, Ysize are the pixel size of the image.
    lw is the line width of the output image strokes.
    Note: This function uses cv2, so import cv2
    Example: >>>a = draw2img(drawings[3], 256, 256)
             >>>plt.imshow(a)
    '''
    fig, ax = plt.subplots()
    drawing = eval(drawing)
    for x,y in drawing:
        ax.plot(x, y,'g',  marker='.', linewidth = lw) #  marker='.', See which line width is better, <4 might be good
    ax.axis('off')
    fig.canvas.draw()    
    X = np.array(fig.canvas.renderer._renderer)
    plt.close(fig)
    # image resizing. Original X is of various size due to strokes variable's length
    temp = (cv2.resize(X, shape) / 255.)[::-1]
    return temp[:,:,1] # only green channel, as we have drawn with green


# In[ ]:


'''
import time
start_time = time.time()
print(f"--- {(time.time() - start_time)} seconds ---") 
'''
# It took me 26 secs to load 1000 *340 images (1000 per class), possibly 100MB of data
train = pd.DataFrame()
for file in os.listdir('../input/train_simplified/'):
    train = train.append(pd.read_csv('../input/train_simplified/' + file, index_col='key_id', nrows=10))


# In[ ]:


train.head(5)


# **Cleaning and mapping** : word column need to be cleaned with '_' and drawing need to be transformed to 32*32 pixels. To save on memory I will remove the orignal drawing. 

# In[ ]:


train['word']= train['word'].apply(lambda x: x.replace(' ', '_'))
train.head(4)
#train['word'].values


# In[ ]:


# most time consuing, part, think of getting rid of it. 
train['drawing'] = train['drawing'].apply(draw2img) # for this training set eval() necessary in draw2img


# In[ ]:


plt.imshow(train['drawing'].values[0])


# So far it is clear that a lot data processing require. And here the size of trainig set is HUGE. It is good for the Neural Nets but constrain is that we don't have enough memory to process them all. Thinking what can be done for this. May be small batch generator function, which will randomly choose a small set of training set and then proces on them before learning anything. I will try this for now.   In doing differet proces steps, we may need to use some of the disk space for svaing some files. In this case /working directory might be helpful.

# **Serious Stuff**

# Following is a list of parameters, by chaning them we can scale up our calculation. I have tested up to 3 classes with large data set. Please change these following parameter as you feel comfortable with your memory. If you find any better solution for memory please share in the comment. 

# In[ ]:


# parameters
shape = (32, 32)
training_classes = 2 # how many class we are training now
train_size = 60 # how many images per class considering for training+evaluation


# In[ ]:


# modifying the drawing to image function
# use it as train['drawing'] = train['drawing'].apply(draw2img)
def draw2img(drawing, shape = shape):
    fig, ax = plt.subplots()
    #drawing = eval(drawing)
    for x,y in drawing:
        ax.plot(x, y,'g',  marker='.') #  marker='.',
    ax.axis('off')
    fig.canvas.draw()    
    X = np.array(fig.canvas.renderer._renderer)
    plt.close(fig)
    # image resizing. Original X is of various size due to strokes variable's length
    temp = (cv2.resize(X, shape) / 255.)[::-1]
    return temp[:,:,1].astype('int8') # only green channel, as we have drawn with green, try bool


# Reading and processing simultaneously. This is to save memory we will selectively read columns from the csv files. And at the same time data type will be assigned as needed. For example default data type for number is float64 which cost 8 bytes. Just to save a non negative integer of the range 0 to 255 we need int8. So reading and saving only required column will give better edge on memory. 

# In[ ]:


import time
start_time = time.time()

train = pd.DataFrame()
i = 0
labels = dict()
for file in os.listdir('../input/train_simplified/'):
    print(f"Reading...{file}.....{i*100/340}% complete")
    label = file.split('.')[0].replace(' ', '_')
    labels.update({i:label})
    
    temp = pd.read_csv('../input/train_simplified/' + file, nrows=train_size, 
                                    usecols = ['drawing', 'word'])
    # processing data
    temp['drawing'] = [ast.literal_eval(pts) for pts in temp['drawing'].values]   
    temp['drawing'] = temp['drawing'].apply(draw2img)
    
    #global label encoding
    temp['word']    = np.int16(i)
    train = train.append(temp)
    
    i = i+1
    if i==training_classes: 
        break
    if i%10==0:
        print(f"Time elasped in reading: {(time.time() - start_time)} seconds ---") 


print(f"Total Time elasped in reading: {(time.time() - start_time)} seconds ---") 

# labesl is the dictionary to convert lebels from number to the actual words
df_labels = pd.DataFrame(index = list(labels.keys()), 
                         data = list(labels.items()),
                         columns = ['Class Code', 'Word'] )
df_labels.to_csv('labels.csv')
del labels


# Preparing data

# In[ ]:


# preparing x_train and y_train
x = np.array(train['drawing'])
y = np.array(train['word'])
# each row of x, y is a input 

# y_train to onehot encoding for making them as useful for output softmax layer
'''
array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
'''
y =y.reshape(-1, 1)  # making it a 2d array like [[1], [1], ]
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc.fit(y)
y = enc.transform(y).toarray()

#del train
gc.collect()

# test train split

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=101)

print("Taking care of dimensions--------------------")
print(f"shape of x_train: {x_train.shape}")
print(f"shape of x_test: {x_test.shape}")
print(f"shape of y_train: {y_train.shape}")
print(f"shape of y_test: {y_test.shape}")
print(f"shape of image: {x_train[1].shape}")


# Genrating class for batch wise processing of data

# In[ ]:


# class for batch processing
class qdHelper():
    def __init__(self):
        self.i= 0
        self.x_train = x_train
        self.x_test = x_test        
        self.y_train = y_train
        self.y_test = y_test
    def setupimage(self):
        self.x_train = np.vstack([a for a in x_train]).reshape(x_train.shape[0],32,32,1)
        self.x_test = np.vstack([a for a in x_test]).reshape(x_test.shape[0],32,32,1)

    def next_batch(self, batch_size):
        x = self.x_train[self.i:self.i+batch_size]
        y = self.y_train[self.i:self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.x_train)
        return x, y


# In[ ]:


# instantiation
qd = qdHelper() 
qd.setupimage()


# **TENSORFLOW VARIABLE AND GRAPH SETUP**

# In[ ]:


import tensorflow as tf
# creating placeholder
x = tf.placeholder(tf.float32, shape = [None, 32,32,1])
y_true = tf.placeholder(tf.int8, shape = [None, training_classes])
hold_prob = tf.placeholder(tf.float32)

# Helper Function
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape = shape)
    return tf.Variable(init_bias_vals)

# 2d conv without with strides 1 pixel in each dimension
# with padding same to keep the image dimension same.
def conv2d(x,W):
    return tf.nn.conv2d(x,W, strides = [1,1,1,1],padding= 'SAME')
# defining a 2 by 2 maximum poooling
def max_pool(x):
    return tf.nn.max_pool(x, ksize= [1,2,2,1],
                         strides = [1,2,2,1],
                         padding = 'SAME')
# this layer is Wx+b , b is of of the last dimension
def convlayer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W)+b)

def full_layer(input_layer, output_size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, output_size])
    b = init_bias([output_size])
    return tf.matmul(input_layer, W)+b

# creating graph
# shape of W, width, height, channel_in, channel_out
conv1 = convlayer(x, shape = [4,4,1,32])
conv1_pool = max_pool(conv1)

# previous layer ouput 32 channels, so it will be the input
# in this layere, and we want 64 output. 
conv2 = convlayer(conv1_pool, shape = [4,4,32,64])
conv2_pool = max_pool(conv2)

# creatiing flattened features
# images got 2 max pool layer of 2 by 2, so got a shape of 32 by 32 now
conv2_flat = tf.reshape(conv2_pool, [-1, 8*8*64])

# fully connected layer
layer1 = tf.nn.relu(full_layer(conv2_flat, 512))
# drop out layer
layer1_drop = tf.nn.dropout(layer1, keep_prob= hold_prob)

y_pred = full_layer(layer1_drop,training_classes )

# loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=y_true,logits=y_pred))

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)


# Computing the Graph

# In[ ]:


# variable initilazation
init = tf.global_variables_initializer()

#Graph session
with tf.Session() as sess:
    sess.run(init)

    for i in range(10000):
        batch = qd.next_batch(100)
        sess.run(train, feed_dict={x: batch[0], y_true: batch[1], hold_prob: 0.5})
        
        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i%100 == 0:            
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))

            acc = tf.reduce_mean(tf.cast(matches,tf.float32))
            print(f"done on {i}")

            print(sess.run(acc,feed_dict={x:qd.x_test,y_true:qd.y_test,hold_prob:1.0}))
            print('\n')              


# This is for today. I will be keep updating this kernel daily basis. Please stay tuned to get update. 
