#!/usr/bin/env python
# coding: utf-8

# * ## This Notebook Demonstrates 'Transfer Learning' in Keras
# 
# * Press "Fork" at the top-right of this screen to run this notebook yourself.
# * Change some values in the fit command
# * Let's see who can achieve the best predictions by class on Monday.
# 
# Thanks to abhiksark for the original version of this notebook:
# https://www.kaggle.com/abhiksark/introduction-to-transfer-learning-cats-dogs

# ## Transfer learning
# It is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task. 
# 
# <img src="https://elearningindustry.com/wp-content/uploads/2016/09/5-tips-improve-knowledge-transfer-elearning-e1475138920743.jpeg" width="400px"/>
# 
# The above image exactly represets what the model does for you.
# It remembers the learning from a fairly related problem and applies it to the problem having new data.
# 
# 
# Pre-trained Model Approach :-
# 
#    1.  Select Source Model. A pre-trained source model is chosen from available models. Many research institutions release models on large and challenging datasets that may be included in the pool of candidate models from which to choose from.
#    <br/><br/>
#    2.  Reuse Model. The model pre-trained model can then be used as the starting point for a model on the second task of interest. This may involve using all or parts of the model, depending on the modeling technique used.  
#    <br/>
#    3.  Tune Model. Optionally, the model may need to be adapted or refined on the input-output pair data available for the task of interest.
#    <br/>
# ![](http://https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/09/Three-ways-in-which-transfer-might-improve-learning.png)    
# 
# Pre-trained Models <br/>
# <a href="https://github.com/KaimingHe/deep-residual-networks">Microsoft ResNet Model</a> <br/>
# <a href="https://github.com/tensorflow/models/tree/master/inception">Google Inception Model</a> <br/>
# <a href="https://github.com/BVLC/caffe/wiki/Model-Zoo">Caffe Model Zoo</a>
# 
# 

# ## Libraries ##
# Importing the Required libraries. libraries allow programmers to share their solutions, so we don't have to keep re-inventing the wheel. While the "language" defines the core requirements of a Turing compete computing device, additional features often emerge as libraries that use the language to perform specific functions. - bonus question: What does turing complete mean?

# In[ ]:


print( "hello world")


# In[ ]:


"hello world"


# In[ ]:


import os, cv2, random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm    #Helps in visualization
from random import shuffle #to shuffle the images 


# ## settings ##
# Our code will often reuse a number of string value, or we may want to change that value depending on circumstances. putting these values near the top and together makes it easier to find and change these values. Imagesize is a good example.

# In[ ]:




TRAIN_DIR = '../input/paper-training-images/TrainingResized/'
TEST_DIR = '../input/paper-training-images/TrainingResized/'
IMG_SIZE = 128  

SHORT_LIST_TRAIN = os.listdir(TRAIN_DIR) #using a subset of data as resouces as limited. 
SHORT_LIST_TEST = os.listdir(TEST_DIR)

SHORT_LIST_TRAIN


# ## Run a command line call ##
# 
# This line is a bit tricky - called a magic function - it's purpose is to set the "inline" flag in the matplotlib already imported into this kernal you can remove it and observe the difference, but you would need to restart the kernal because this "inline" flag is persistant.
# 
# https://stackoverflow.com/questions/43027980/purpose-of-matplotlib-inline
# 
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Methods ##
# "Methods" provide a means of "encapsulating" a series of commands. In this case "label_im" can be used as a short-hand for the trnslation between the words "dog" and "cat", and a numeric representation of 1 and 0 in a "one hot" array.

# In[ ]:


def label_img(img): 
    if "Food container" in img: 
        return [1,0,0,0]
    elif "food tray" in img: 
        return [0,1,0,0]
    elif "paper cup" in img: 
        return [0,0,1,0]
    elif "paper plate" in img: 
        return [0,0,0,1]
    


# In[ ]:


#returns an numpy array of train and test data
def create_train_data():
    training_data = []
    for img in tqdm(SHORT_LIST_TRAIN):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    shuffle(testing_data)
    return testing_data


# ## Visualizing Classes ##
# Visulization is how we interact with complex datasets - we measure and sample them in many ways.
# 

# In[ ]:



labels = [] #make an empty list.
for i in SHORT_LIST_TRAIN: 
    labels.append(str(label_img(i)))

sns.countplot(labels) # show the list as a graph
plt.title('Recycling Classes')


# ## Creating a Training Set Data##

# In[ ]:


train = create_train_data() #This is a method defined above


# Lets have a quick look at some of the training images

# In[ ]:



#import matplotlib.pyplot as plt

fig , ax = plt.subplots(3, 3, figsize=(30, 25))
for i, axis in enumerate(ax.flat):
    axis.imshow(train[i][0], cmap='gray')
    axis.set_title(f'Label:  {train[i][1]}', fontsize=20)


# ## Specify Model##
# 
# This code will result in a deperecation warning.
# Things change, and in software, when the developers of the language realize that some feature is causing problems, they may choose to provide a different features, in which case, they will create a warning message to people using the old feature, and then, in a yet later version, they may remove the offending feature completely.

# In[ ]:


from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

NUM_CLASSES = 4
RESNET_WEIGHTS_PATH = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5' #importing a pretrained model
my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='max', weights=RESNET_WEIGHTS_PATH))
my_new_model.add(Dense(NUM_CLASSES, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = True


# In[ ]:


my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# ## Compile Model##

# In[ ]:


my_new_model.summary()


# ## From Train Dividing X and Y##
# because "fit" requires labels and data to be in seperated lists, we seperate the lists here.
# 

# In[ ]:


X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = np.array([i[1] for i in train])


# ## Fit Model##
# 
# this is machine learing!
# 
# https://keras.io/models/sequential/#fit
# Trains the model for a given number of epochs (iterations on a dataset).
# 
# Returns
# 
# A History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
# 
# 

# In[ ]:


history = my_new_model.fit(X, Y, validation_split=0.20, epochs=4, batch_size=64)


# ## Plotting loss and accuracy for the model##
# 
# now we're going to look at the history object returned by Training

# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ## Repeat ##
# Here it all is again in one section:
# Create a list,
# break it X&Y
# Train - this time with different values
# And plot the learning curve so we can compare how changing the values affects learning
# 
# Discussion: What Values are different in this test vs. the previous test?
# 

# In[ ]:



SHORT_LIST_TRAIN = os.listdir(TRAIN_DIR) #[0:500]
train = create_train_data()
X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = np.array([i[1] for i in train])
history = my_new_model.fit(X, Y, validation_split=0.5, epochs=20, batch_size=64)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ## Use the model to predict the labels of some Images ##
# 

# In[ ]:


testing_data = []
for img in tqdm(os.listdir(TEST_DIR)[0:100]):
    path = os.path.join(TEST_DIR,img)
    img_num = img.split('.')[0]
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    testing_data.append([np.array(img), img_num])
    
shuffle(testing_data)    
test_data = testing_data 


fig , ax = plt.subplots(6, 4, figsize=(30, 25))
for i, axis in enumerate(ax.flat):
    axis.imshow(test_data[i][0], cmap='gray')
    img_data = test_data[i][0]
    orig = img_data
    data = img_data.reshape(-1,IMG_SIZE,IMG_SIZE,3)
    model_out = my_new_model.predict([data])[0]    

    #axis.set(title=f'{im_pred[i].max()} => {category[im_pred[i].argmax()]}')
    axis.set_title(f'Predict: {model_out.max()} => {model_out.argmax()}', fontsize=20)


# ## Sources
# 1. https://machinelearningmastery.com/transfer-learning-for-deep-learning/ <br/>
# 2. http://cs231n.github.io/transfer-learning/ <br/>
# 3. https://arxiv.org/abs/1411.1792 <br/>
# If you are Interested in Research.
# 
# 
# 

# Please upvote this kernel so that it reaches the top of the chart and is easily locatable by new users. Your comments on how we can improve this kernel is welcome. Thanks You For your Support.
