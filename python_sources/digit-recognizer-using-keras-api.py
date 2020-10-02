#!/usr/bin/env python
# coding: utf-8

# **Digits Recognizer Using keras**
# <hr/>
# This notebook is inspired by Andrew Ng's CNN course from the Deep Learning Specialization track on Coursera and a WiMLDS meetup
# 
# I stumbled on the Kaggle Digit Recognizer [Competition](https://https://www.kaggle.com/c/digit-recognizer)  and decided to try it out using the MNIST dataset provided.
# 
# its a digit recognizer trained on the MNIST digits dataset. I built it on the Keras API. I initially tested tried it on my CPU, but it took too long to train a single epoch. Got introduced to Google Colab Notebook... (Ended up being a life saver).
# 
# If you will run this notebook, on CPU, i'll recommend you set the epoch to 3. If you have a GPU, you are good to run the entire 20 epochs.
# 
# PS: Created the notebook on Google's Colab, so you may see some google packages... You can try it out!

# In[ ]:


#lets import the required packages

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pandas as pd

import tensorflow as tf
from keras import Sequential, Input, Model
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.layers import  Dense, Dropout, Flatten, Conv2D,MaxPooling2D
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam, RMSprop,Adadelta, Adagrad
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import Softmax,LeakyReLU,activations
from sklearn.model_selection import train_test_split


# In[ ]:


#loading the datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


#seperate the label into a variable y_train, and drop the column from our training set
y_train = train['label']
x_train = train.drop(['label'], axis =1) 
x_test = test


# # Dataset Inspection
# Lets inspect the shape of the data

# In[ ]:


#Lets inspect the shape of the dataset
print('Train X  Dimension :',x_train.shape)
print('Test Dimension :',x_test.shape)


# There are 42K  rows and 785 columns for training, while test has 28K rows and 748 columns.

# **Check both datasets for null or missing values** 

# In[ ]:


print(x_train.isnull().any().describe())


# In[ ]:


print(x_test.isnull().any().describe()) 


# In[ ]:


#Are the categories well represented ?
sns.set(style="darkgrid")
sns.countplot(y_train)
plt.xlabel(' Digits')
plt.ylabel('Count')


# Yes, they are. The categories are well represented

# 
# 

# In[ ]:


#Check for what categories the label contains
categories = y_train.unique()
print('Output Categories: ',categories)
print('Total number of  Categories: ',len(categories))


# In[ ]:


#categories


# **Fair Enough no problems.. **Dont expect your datasets to be this neat all the  time. Time to treat the data before we can feed it into a CNN

# In[ ]:


#x_train = (x_train.iloc[:,1:].values).astype('float32') # all pixel values
#y_train = y_train.values.astype('int32') # only labels i.e targets digits
#x_test = x_test.values.astype('float32')


# # Dataset Preparation
# Before we design the  Convultion Network, its important to treat the dataset. Various steps are involved.
# * Normalization
# * Reshaping
# * Label Encoding
# * Splitting data into Training and Validation sets
# 

# ## Normalization
# Here we performa grayscale normalization on the data, technically we are centering the data around zero mean and unit variance, mean = 0, variance =1

# In[ ]:


X_train = x_train/255.0
X_test = x_test/255.0


# In[ ]:


x_train.shape


# ## Reshaping
# Here we reshape the image into a 3 dimension matrices of  28px by 28px by 1

# In[ ]:


x_train = X_train.values.reshape(-1,28,28,1)
x_test = X_test.values.reshape(-1,28,28,1)


# ## Label Encoding
# A one-hot vector is a vector which is 1 in an single dimension and zeros elsewhere, for instance 6 will be [0,0,0,0,0,0,1,0,0,0]
# 

# In[ ]:


y_train_enc = to_categorical(y = y_train, num_classes= len(categories))
# Display the change for category label using one-hot encoding
print('Original label:', y_train[25])
print('After conversion to one-hot:', y_train_enc[25])


# * ### Lets Visulaize some of the images

# In[ ]:


for i in range(0, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(x_train[i][:,:,0], cmap=plt.get_cmap('gray'))
    plt.title(y_train[i]);


# In[ ]:


seed = 30
np.random.seed(seed)


# # Spliting Data into Training and Validation Set
# I'll split the training data into two parts; a small percentage - 15% to contain the validation set, on which we would evaluate the model, while the data will be trainined on the remaining 85%

# In[ ]:


#Spliting the Data
train_x, validation_x, train_y,validation_y = train_test_split(x_train,y_train_enc, test_size = 0.15, random_state = seed)


# # Model Architecture
# I'll use the [Keras](https://https://keras.io/) Sequential API, where I'll have to create the CNN one step at a time by adding layers. 
# 
# The first layer is the Convultional2D layer, which is a set of learnable filters, i want to model to learn,. The first Conv2D latyer will contain 32 filters, the next will conv layer will contain 64 filter,  wile the last Conv2D layer sill have 128 filters.
# Inbetween the Conv2D layers there will be be an Activation Layer, 
# 
# I've chosen the Rectified Linear Unit, this activatinon function adds some non linearity to the model.
# 
# After the activation layer is a MaxPooling2D Layer, this layer acts a as a downsampling filter, it selects the maximum pixel in a block defined by the filter parameter i specified. It looks at the two neigbhouring pixels and pick the maximum pixel. Other implementations of the of this feature include [Average Pooling](https://keras.io/layers/pooling/#averagepooling2d)  and [GlobalPooling2D](https://keras.io/layers/pooling/#globalmaxpooling2d). This MaxPooling reduces computational cost.
# 
# After the Pooling Layer is the DropoutLayer, this is used for regularization. It randomly drops some nodes in the layer.
# 
# The Flatten layer converts the features into a single 1D vectors and finally the Dense Layer which is uses the softmax classifier

# In[ ]:


import keras
from keras import Sequential, Input, Model
from keras.layers import  Dense, Dropout, Flatten, Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import Softmax,LeakyReLU,activations
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, RMSprop, SGD


# In[ ]:


#some HyperParameters. Feel Free to tune them
batch_size = 128
epochs = 20
alpha = 0.3
num_classes = 10


# In[ ]:


#Model Architecture in Summary is [[Conv2D -> ReLU -> MaxPool2D -> DroupOut]] *2 -> Dense -> ReLU -> Flatten -> Droupout -> Dense -> Out
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='linear', padding='same',input_shape = (28,28,1)))
model.add(LeakyReLU(alpha=alpha))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Dropout(rate = 0.5))

model.add(Conv2D(64, kernel_size=(3,3), activation='linear', padding='same'))
model.add(LeakyReLU(alpha=alpha))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Dropout(rate = 0.4))

model.add(Dense(128, activation='relu'))
model.add(LeakyReLU(alpha=alpha))
model.add(Flatten())

model.add(Dropout(rate=0.4))
model.add(Dense(len(categories), activation='softmax'))
model.compile(loss=categorical_crossentropy, optimizer=Adagrad(), metrics=['accuracy'])
model.summary()


# In[ ]:


#training the model
model_train = model.fit(x = train_x, y = train_y, batch_size= batch_size, epochs = epochs, validation_data=(validation_x,validation_y))


# Lets inspect the Model if its overfitting

# In[ ]:


#extracting the training history params. this will give some information if its overfitting
train_acc = model_train.history['acc']
train_loss = model_train.history['loss']
val_acc = model_train.history['val_acc']
val_loss = model_train.history['val_loss']

ep = range(len(train_acc))
plt.plot(ep, train_acc,  label='Training accuracy', color ='g')
plt.plot(ep, val_acc, 'b', label='Validation accuracy',color='r')
plt.title('Training and validation accuracy')
plt.legend()


# Our Model isnt doing bad afterall. Validation Accuracy is higher than the training accuracy alomst everytime while training. This means the model isnt oferfitting
# 
# ### Feel free to tune the hperparameters, modify the NN architecture or try different optimizers

# In[ ]:


#plt.figure()
plt.plot(ep, train_loss, label='Training loss')
plt.plot(ep, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# The validation loss is reducing after almost every epoch, this clearly indicates the model isnt doing bad

# # Prediction and Confusion Matrix
# 
# Lets Make predictions from the Model and visualize the confusion matrix

# In[ ]:


#Lets use our model to predict
y_pred = model.predict(validation_x,verbose =1)

#Convert predicted category to one hot vectors
y_pred_class = np.argmax(y_pred, axis =1 )

valid_y_class = np.argmax(validation_y, axis = 1)
#lets generate the confusion matrix so we can see how right the predictions are
confuse_matrix = confusion_matrix(valid_y_class,y_pred_class)

plt.figure(figsize = (10,10))
sns.heatmap(confuse_matrix, annot= True, fmt = 'd', cmap = 'YlGnBu', linewidths=.9,linecolor='black')
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')


# ## Why is our model classifying some values wrongly
# 
# Perhaps some visual inspections  will tell us why
# 
# Classification report will help us identifying the misclassified digits in more detail. We will be able to observe the model performance and  identify which classes it performed poorly

# In[ ]:


from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(valid_y_class, y_pred_class, target_names=target_names))


# The Model is doing pretty well!

# In[ ]:


errors = (y_pred_class - valid_y_class != 0)
y_pred_classes_errors = y_pred_class[errors]
y_pred_errors = y_pred[errors]
y_true_errors = valid_y_class[errors]
x_val_errors = validation_x[errors]


# The displayerror() function is inspired by [Yassineghouzam](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6)

# In[ ]:


def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1


# In[ ]:


# Probabilities of the wrong predicted numbers
y_pred_errors_prob = np.max(y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, x_val_errors, y_pred_classes_errors, y_true_errors)


# ** The erros are obvious afterall..**

# In[ ]:


#Lets use the model to predict the testset from Kaggle Competition
predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis = 1)
predictions = pd.Series(predictions, name =  'Label')


# In[ ]:


submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)


# # Training on Entire Competition Training set
# After you've evaluated the model, and confim its doing well, you can then proceed to use the competition's training set and predict on the test set.

# In[ ]:


#training the model
model_train_full = model.fit(x = x_train, y = y_train_enc, batch_size= batch_size, epochs = epochs)


# In[ ]:


#Lets use the model to predict the testset from Kaggle Competition
predictions_full = model.predict(x_test)
predictions_full = np.argmax(predictions_full, axis = 1)
predictions_full = pd.Series(predictions_full, name =  'Label')
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions_full)+1)),"Label": predictions})
submissions.to_csv("Submission.csv", index=False, header=True)


# #### if you find this note useful, some upvotes will be appreciated.. Feel free to criticize the notebook. 
# 
# PS: This is my first kernel here... 
