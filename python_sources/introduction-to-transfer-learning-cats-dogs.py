#!/usr/bin/env python
# coding: utf-8

# ## This Notebook Introduces How to apply 'Transfer Learning' in Kaggle
# 
# Thank you for opening this Notebook!
# 
# Press "Fork" at the top-right of this screen to run this notebook yourself and build each of the examples.
# 
# I have made all efforts to document each and every step involved so that this notebook acts as a good starting point for new Kagglers who hope to apply **Transfer Learning** to their problem.

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

# ## Libraries and settings ##
# Importing the Required libraries and assignment of few constants(such as Directories)

# In[ ]:


import os, cv2, random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm    #Helps in visualization
from random import shuffle #to shuffle the images 

get_ipython().run_line_magic('matplotlib', 'inline')

TRAIN_DIR = '../input/dogs-vs-cats-redux-kernels-edition/train/'
TEST_DIR = '../input/dogs-vs-cats-redux-kernels-edition/test/'
IMG_SIZE = 224

SHORT_LIST_TRAIN = os.listdir(TRAIN_DIR)[0:10000] #using a subset of data as resouces as limited. 
SHORT_LIST_TEST = os.listdir(TEST_DIR)


# In[ ]:


def label_img(img): 
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat': return [1,0]
    #                             [no cat, very doggo]
    elif word_label == 'dog': return [0,1]


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
# Visulization is important to compare differnet classes and their number of occurances.

# In[ ]:



labels = []
for i in SHORT_LIST_TRAIN:
    if 'dog' in i:
        labels.append(1)
    else:
        labels.append(0)

sns.countplot(labels)
plt.title('Cats and Dogs')


# ## Creating a Training Set Data##

# In[ ]:


train = create_train_data()


# ## From Train Dividing X and Y##

# In[ ]:


X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = np.array([i[1] for i in train])


# ## Specify Model##

# In[ ]:


from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

NUM_CLASSES = 2
RESNET_WEIGHTS_PATH = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5' #importing a pretrained model
my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='max', weights=RESNET_WEIGHTS_PATH))
my_new_model.add(Dense(NUM_CLASSES, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = True


# ## Compile Model##

# In[ ]:


my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# ## Model Sumary##

# In[ ]:


my_new_model.summary()


# ## Fit Model##

# In[ ]:


history = my_new_model.fit(X, Y, validation_split=0.20, epochs=4, batch_size=64)


# ## Plotting loss and accuracy for the model##

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


# In[ ]:



SHORT_LIST_TRAIN = os.listdir(TRAIN_DIR)[-5000:]
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


# ## Testing Model on the Test Data##

# In[ ]:


import matplotlib.pyplot as plt

# if you need to create the data:
test_data = process_test_data()
# if you already have some saved:
#test_data = np.load('test_data.npy')

fig=plt.figure()

for num,data in enumerate(test_data[:12]):
    # cat: [1,0]
    # dog: [0,1]
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(-1,IMG_SIZE,IMG_SIZE,3)
    #model_out = model.predict([data])[0]
    model_out = my_new_model.predict([data])[0]
    
    if np.argmax(model_out) == 1: str_label='Dog'
    else: str_label='Cat'
        
    y.imshow(orig)
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()


# In[ ]:


prob = []
img_list = []
for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(-1,IMG_SIZE,IMG_SIZE,3)
        model_out = my_new_model.predict([data])[0]
        img_list.append(img_num)
        prob.append(model_out[1])


# In[ ]:


submission = pd.DataFrame({'id':img_list , 'label':prob})
print(submission.head())
submission.to_csv("../working/submit.csv", index=False)


# ## Sources
# 1. https://machinelearningmastery.com/transfer-learning-for-deep-learning/ <br/>
# 2. http://cs231n.github.io/transfer-learning/ <br/>
# 3. https://arxiv.org/abs/1411.1792 <br/>
# If you are Interested in Research.
# 
# 
# 

# Please upvote this kernel so that it reaches the top of the chart and is easily locatable by new users. Your comments on how we can improve this kernel is welcome. Thanks You For your Support.
