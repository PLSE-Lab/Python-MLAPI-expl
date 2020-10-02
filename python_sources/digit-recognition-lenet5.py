#!/usr/bin/env python
# coding: utf-8

# # Digit Recognizer
# <html>
#     <body>
# <p><t>There are 2 files. They are <li>train.csv</li> <li>test.csv</li> <p>which are used for training and testing respectively. <br>Each row in dataset contain <b>28 x 28 gray-scale image</b> of hand-drawn digit flattened as <b>784 pixels</b>.This pixel values from 0 to 255. These number are from <b>zero to nine</b> indicated by "label" column in train dataset.</p>
# <p> This kernel consists of 4 sections<p>
#     <ol>
#         <li> Data Analysis</li>
#         <li> Data preprocessing</li>
#         <li> Training</li>
#         <li> Testing</li>
#     </ol>
# <p>Dataset consists of <ul><li>Train data 42,000 rows</li> <li>Test data 28,000 rows</li></ul>
# 
# The train dataset is divided into training (90%) and development (10%) datasets.
# 
# This data is training on LeNet-5 model which have 2 Convolution layers and 3 fully connected layers published this <a href = "http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf"> paper</a> by Yann LeCun 1998 who received Turning award in 2019.<br><br>
# <b>Applied a trick to make Lenet architecture comfortable with these data dimensions. Let's see what is that!
#     </body>
# </html>

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, AveragePooling2D, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

random_seed = 1
np.random.seed(random_seed)


# # Data Analysis

# In[3]:


# loading training data
train = pd.read_csv('../input/train.csv')
train.shape


# In[4]:


# loading test data
test = pd.read_csv('../input/test.csv')
test.shape


# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


# Dividing Independent and Dependent features in dataset
Y = train["label"].values
X = train.drop(["label"],axis=1).values


# In[8]:


# Dimensions of each example (a gray image with 28 x 28 is flattern into 784)
X[0].shape


# In[9]:


# Visualizing first example in dataset
plt.imshow(np.reshape(X[0],(28,28)))
plt.title(Y[0])
plt.show()


# In[10]:


# Plotting count of each digit
sns.countplot(Y)


# In[11]:


# Count of each digit
train["label"].value_counts()


# In[12]:


# Checking null values
train.isnull().sum().sum()


# In[13]:


# Plotting grid of first 12 examples
plt.figure(figsize=(10,6))
for i in range(12):
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np.reshape(X[i],(28,28)))
    plt.title(Y[i])
 
plt.show()


# # Data preprocessing

# In[14]:


# Normalizing pixel values from (0 to 255) to (0 to 1)
X = X/255
test /= 255
#Y is digit label, No need of normalization and Test data do not have labels.


# In[15]:


# Checking normalized values
print(X.max(), X.min())


# In[16]:


# Reshaping images
X = X.reshape(-1,28,28,1)  # training examples
test = test.values.reshape(-1,28,28,1) #test examples


# In[17]:


# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y = to_categorical(Y, num_classes = 10)


# In[18]:


# Spliting into Train dataset and Validating(development dataset)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.1, random_state=random_seed)


# In[19]:


# Checking shape of train data
X_train.shape


# ### Now data is ready to train, Let's design our model

# In[20]:


# Defining LeNet model
def lenet5():
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(5,5), strides=(1,1),padding='same', activation='relu', name='Conv2D0', input_shape=(28,28,1)))
    model.add(BatchNormalization(axis=-1, name='bn0'))
    model.add(AveragePooling2D())
    
    model.add(Conv2D(filters=16, kernel_size=(5,5), strides=(1,1), activation='relu', name='Conv2D1'))
    model.add(BatchNormalization(axis=-1, name='bn1'))
    model.add(AveragePooling2D())
    
    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    return model


# In[21]:


# Initializing lenet model and its data flow
model = lenet5()
model.summary()


# # Train our model

# In[22]:


# Compiling our model and training on train data
model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=["accuracy"])
histroy = model.fit(X_train, Y_train, epochs=5, batch_size=16)


# In[23]:


# Plotting Training accuracy and Training Loss curves
train_accuracy = histroy.history['acc']
train_loss = histroy.history['loss']

iterations = range(len(train_accuracy))
plt.plot(iterations, train_accuracy, label='Training accuracy')
plt.title('epochs vs Training accuracy')
plt.legend()

plt.figure()
plt.plot(iterations, train_loss, label='Training Loss')
plt.title('epochs vs Training Loss')
plt.legend()


# # Testing our model

# In[24]:


# Evaluating our model with dev dataset
preds = model.evaluate(x=X_val, y=Y_val)

print ("\nLoss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# In[25]:


# Look at confusion matrix 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 


# In[26]:


model.save("DigiModel.h5")


# In[ ]:


# Predicting on Test data
y_pred = model.predict(test)


# In[ ]:


# Checking sample submission format
sam_sub = pd.read_csv('../input/sample_submission.csv')
sam_sub.head()


# In[ ]:


# Compressing prediction to labels (ex :  [0,0,1,0,0,0,0,0,0,0] -> 2)
y_pred =  np.argmax(y_pred,axis = 1)


# In[ ]:



# Making our predictions as Pandas Series 
y_pred = pd.Series(y_pred,name="Label")
y_pred.shape


# In[ ]:


# Saving our model prediction of test data
# step 1 -> Create a DataFrame with ImageId(given in test data) and Label (Model predicted value)
# step 2 -> Save that DataFrame to make submission in kaggle competition for evaluation
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),y_pred],axis = 1)

submission.to_csv("lenet5_mnist.csv",index=False)


# # Conclusion:
# 1. Applied a Same convolution in First layer because given image size is 28 x 28 where Lenet designed for 32 x 32 input size.
# 2. Trained 5 epochs with 16 batch size
# 3. Achieved 98% accuracy
