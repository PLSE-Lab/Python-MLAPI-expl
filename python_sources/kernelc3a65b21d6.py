#!/usr/bin/env python
# coding: utf-8

# **In this kernel we will experiment a few different ways of image classification technique.**
# *     **Image Classification using traditional image processing and openCV**
# *     **Image Classification using Deep Learing and Keras** 
# *     **Image Classification using Deep Learning and PyTorch**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
random.seed(11112111311114)
# Keras libs import for preprocesssing, objects to build model will be imported later
from keras.preprocessing import image
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
# visualisation libs import
import matplotlib.pyplot as plt
from sklearn import metrics
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Sample images in the Data-set**

# In[ ]:


pos_img_path = "/kaggle/input/cell_images/cell_images/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_162.png"
neg_img_path = "/kaggle/input/cell_images/cell_images/Uninfected/C100P61ThinF_IMG_20150918_144104_cell_128.png"
pos_img = image.load_img(pos_img_path, target_size=(224, 224, 1))
neg_img = image.load_img(neg_img_path, target_size=(224, 224, 1))
x = image.img_to_array(pos_img)
y = image.img_to_array(neg_img)
print("test result evidence: positive")
plt.imshow(x/255.)
plt.show()
print("test result evidence: negative")
plt.imshow(y/255.)
plt.show()


# **Our Aim is to classify the above two images thus detecting weather the cell is infected with malaria or not.**

# In[ ]:


extension = "png"

positive_img_set = []
positive_set_folder = "/kaggle/input/cell_images/cell_images/Parasitized/"
for roots,dir,files in os.walk(positive_set_folder+"."):
    positive_img_set = list(map(lambda x:image.img_to_array(
        image.load_img(positive_set_folder+x, target_size=(224, 224)))/255.,  ##watch out for this normalisation value 255. its really important
                                filter(lambda x: x.endswith('.' + extension),files[:2000])))
    random.shuffle(positive_img_set)
print(len(positive_img_set))

negative_img_set = []
negative_set_folder = "/kaggle/input/cell_images/cell_images/Uninfected/"
for roots,dir,files in os.walk(negative_set_folder+"."):
    negative_img_set = list(map(lambda x:image.img_to_array(
        image.load_img(negative_set_folder+x, target_size=(224, 224)))/255., ##watch out for this normalisation value 255. its really important
                                filter(lambda x: x.endswith('.' + extension),files[:2000])))
    random.shuffle(negative_img_set)
print(len(negative_img_set))
positive_test_set = positive_img_set[1800:]
negative_test_set = negative_img_set[1800:]
positive_train_set =  positive_img_set[:1800]
negative_train_set = negative_img_set[:1800]

train_X = positive_train_set + negative_train_set
train_y = [1]*len(positive_train_set) + [0]*len(negative_train_set)
temp = list(zip(train_X,train_y))
random.shuffle(temp)
train_X,train_y = np.array([x[0] for x in temp]), np.array([x[1] for x in temp])

#del temp
del positive_img_set
del negative_img_set


# In[ ]:


print(train_y)
print([[1,0] if int(x) is 1 else [0,1] for x in train_y])


# **Let's start with Keras ease of POC. Just a few lines and you can have your eureka moment!!!**

# In[ ]:


#firsly the imports
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.losses import binary_crossentropy
from keras.optimizers import SGD, Adam

#create model
model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=((224,224,3)),padding='same'))
model.add(Conv2D(64, kernel_size=3, activation='relu',  padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#optimizer inilisation
adam = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#sgd = SGD()

#compile model using accuracy to measure model performance
model.compile(optimizer=adam, loss=binary_crossentropy, metrics=['accuracy'])

#train the model
model.fit(train_X, train_y,batch_size=32,epochs=150,validation_split=0.2)


# In[ ]:





# In[ ]:


#generate metrics on the above model
#positive_accuracy = len(list(filter(lambda x: x>0.5,model.predict(np.array(positive_test_set)))))/len(positive_test_set)
#negative_accuracy = len(list(filter(lambda x: x<0.5,model.predict(np.array(negative_test_set)))))/len(negative_test_set)
y_true = [1]*len(positive_test_set)+[0]*len(negative_test_set)
y_predicted = np.append(model.predict(np.array(positive_test_set)),model.predict(np.array(negative_test_set)))
print(y_predicted)
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_predicted, pos_label=2)
#print(positive_accuracy, negative_accuracy)
print(fpr,tpr,thresholds)


# **Now lets work with pytorch which is about as easy to write as the above keras implementation and good point its all in python :) **

# In[ ]:


import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
    return(output)

model = torch.nn.Sequential(
          torch.nn.Conv2d(3,64,3,padding=2),
          torch.nn.ReLU(),
          torch.nn.Conv2d(64,64,3,padding=2),
          torch.nn.ReLU(),
          torch.nn.MaxPool2d(2,2),
          torch.nn.Conv2d(64,128,3),
          torch.nn.ReLU(),
          torch.nn.Conv2d(128,128,3),
          torch.nn.ReLU(),
          torch.nn.MaxPool2d(2,2),
          
        )
FC1 = torch.nn.ReLU(torch.nn.Linear(55*55*128,512))
FC2 = torch.nn.ReLU(torch.nn.Linear(512,256))
FC3 = torch.nn.ReLU(torch.nn.Linear(256,2))
FC4 = torch.nn.Softmax(FC3)
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
X = DataLoader([(torch.tensor(np.transpose(x[0],(2,0,1))),torch.tensor(x[1])) for x in temp], batch_size=1,num_workers=4)
print(X)
criterion = torch.nn.CrossEntropyLoss()
for t in range(150):
    #for j in range(len(train_X)):
    for j, data in enumerate(X,0):
      optimizer.zero_grad()
      y_pred = FC3(FC2(FC1(model(data[0]).flatten())))
      print(torch.tensor(data[1],dtype=torch.float32))
      print(y_pred.view(1,2))
      loss = criterion(y_pred.view(1,2), torch.tensor(data[1],dtype=torch.long))
      print(t, loss.item())
      loss.backward()
      optimizer.step()


# Now as you can see, its a little more work we need to do with pytorch, But at the same time wit provides much more flexibility. Why? Think when you are building an arachitecture where the model gets divided into two parallel path for example while training for multiple task, Pytorch gives much more flexibility then keras.

# But in the above cell, One key thing is missing, mini batches, you need to see that I have given the mini batch count =1 (Try changing it to 2 and see what happens). The problem is there isn't any Flatten() layer provided by pytorch alone. you can do it to a tensor but it cannot be a part of architecture. Hence when you pass a batch tensor.flatten(), it wiill flatten all the batches into a 1D matrix. Hence to be able to use mini_batches we need to write code in the following fashion.
# 

# In[ ]:


import torch
import time
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PrepareData(Dataset):
    
    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    

class simpleCNN(torch.nn.Module):
    def __init__(self):
        super(simpleCNN, self).__init__()
        
        #Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(3,64,3,padding=2)
        self.conv2 = torch.nn.Conv2d(64,64,3,padding=2)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.conv3 = torch.nn.Conv2d(64,128,3)
        self.conv4 = torch.nn.Conv2d(128,128,3)
        self.FC1 = torch.nn.Linear(55*55*128,512)
        self.FC2 = torch.nn.Linear(512,256)
        self.FC3 = torch.nn.Linear(256,2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1,55*55*128)
        x = F.relu(self.FC1(x))
        x = F.relu(self.FC2(x))
        x = F.relu(self.FC3(x))
        x = F.softmax(x)
        return(x)

cnn = simpleCNN()
cnn.cuda()
learning_rate = 1e-4
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
ds = PrepareData(np.array([np.transpose(x,(2,0,1)) for x in train_X]),np.array(train_y))
X = DataLoader(ds, batch_size=32,num_workers=10)
print(X)
criterion = torch.nn.CrossEntropyLoss()
for t in range(150):
    verbose = "Epoch:{}, Loss of last element:{}"
    temp_loss = None
    tm = time.time()
    for j, data in enumerate(X,0):
      optimizer.zero_grad()
      y_pred = cnn(data[0].cuda())
      loss = criterion(y_pred, torch.tensor(data[1],dtype=torch.long).cuda())
      temp_loss = loss.item()
      loss.backward()
      optimizer.step()
      if (j%100==0):
          print(j)
    print(verbose.format(t,temp_loss))
    print("this epoch took:",str(time.time()-tm))
    


# In[ ]:


positive_test_tensors = torch.tensor(list(map(lambda x: np.transpose(np.array(x),(2,0,1)),positive_test_set)))
negative_test_tensors = torch.tensor(list(map(lambda x: np.transpose(np.array(x),(2,0,1)),negative_test_set)))
y_true = [1]*len(positive_test_set)+[0]*len(negative_test_set)
y_predicted = np.append(cnn(positive_test_tensors.cuda()),cnn(negative_test_tensors.cuda()))
print(y_predicted)
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_predicted, pos_label=2)
#print(positive_accuracy, negative_accuracy)
print(fpr,tpr,thresholds)


# ## Next is Tensorflow :) 

# Now you will see the ancient one (ok not ancient but still it got the feels so bear with me, I am doing this for you). Keras is actually the layer over this. Tensorflow is actually built in c/c++ with seamless integration wth python (ok not so seamless but good enough). Now if keras is already built, u should ask why tensorflow, why dont we forget it like we forget what goes on behind pythons sort function ( not that I have forgotten i remember that language on the tip of my fingers! all hail coreman). The point is Keras is great for POC(s) and fast experimentation but when you think production worthy Keras might hold u onto it. Tensorflow gives you flexibility to modify and optimize according to ur requirements and costs

# In[ ]:


import tensorflow as tf
import time
class simpleCNN_tf:
    def __init__(self):
        return
    
    def build(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue,
            green,
            red
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, [5, 5, 3, 64], "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, [5, 5, 64, 64],"conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')
        self.conv2_1 = self.conv_layer(self.pool1, [5, 5, 64, 64], "conv1_1")
        self.conv2_2 = self.conv_layer(self.conv2_1,[5, 5, 64, 64],"conv1_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool1')
        self.fc6 = self.fc_layer(self.pool2,512)
        assert self.fc6.get_shape().as_list()[1:] == [512]
        self.relu6 = tf.nn.relu(self.fc6)
        self.fc7 = self.fc_layer(self.relu6,256)
        self.relu7 = tf.nn.relu(self.fc7)
        self.fc8 = self.fc_layer(self.relu7,2)
        self.relu8 = tf.nn.relu(self.fc8)
        self.prob = tf.nn.softmax(self.relu8, name="prob")
        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom,filter_dim,name):
        with tf.variable_scope(name):
            kernel = tf.Variable(tf.random_normal(filter_dim))
            conv = tf.nn.conv2d(bottom, kernel, [1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(conv)
            return relu

    def fc_layer(self, bottom, output_dim):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])
        fc = tf.layers.dense(x, output_dim)
        return fc

    def get_conv_filter(self, name):
        return tf.layers.dense(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.layers.dense(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")


# In[ ]:



train_y_tf = [[1,0] if int(x) is 1 else [0,1] for x in train_y]

batch_size = 32
epochs = 150
print(len(train_X))
with tf.device('/gpu:0'):
    sess = tf.Session()

    images = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [batch_size, 2])
    train_mode = tf.placeholder(tf.bool)

    cnn = simpleCNN_tf()
    cnn.build(images)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    #print(vgg.get_var_count())

    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        i=0
        a = time.time()
        while(i<len(train_X)):
            print(epoch)
            if len(train_X)-i < batch_size:
                i = i+batch_size
                continue
            batch = train_X[i:i+batch_size]
            true_val = train_y[i:i+batch_size]
            i = i+batch_size
            prob = sess.run(cnn.prob, feed_dict={images: batch, train_mode: True})
            #print(prob)
            cost = tf.losses.softmax_cross_entropy(true_out,cnn.prob)
            train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
            sess.run(train, feed_dict={images: batch, true_out: true_val, train_mode: True})
        print(epoch)
        print(time.time()-a)
            
        

