#!/usr/bin/env python
# coding: utf-8

# # Classifying hand drawn images into the digits they denote
# ## An introduction to Keras using one of the most famous image classification problems
# 
# **Please upvote if you find this kernel useful!**

# ## Library imports

# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from keras.utils import to_categorical
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten, Dropout
from matplotlib import pyplot as plt
import pandas as pd


# ## Loading the dataset

# In[ ]:


traindf = pd.read_csv("../input/train.csv")
testdf = pd.read_csv("../input/test.csv")


# In[ ]:


print(traindf.shape)
traindf.head()


# In[ ]:


print(testdf.shape)
testdf.head()


# There are 10 classes to classify the images into (0-9)

# In[ ]:


train_lbls = traindf.label
print(len(train_lbls))
print(len(np.unique(train_lbls)))


# Shape of the test and train labels to see how many test and train images we have

# In[ ]:


X_train = traindf.drop(labels = ["label"],axis = 1)
print(X_train.shape)


# We have 42,000 images, each one is 28x28 

# Converting values to lie between 0 and 1 (since they are currently digits between 0 and 255, denoting how black each pixel is)

# In[ ]:


X_train = X_train / 255.0
test = testdf.as_matrix() / 255.0


# In[ ]:


test_data = test.astype('float32')


# Reshaping the training and test data so the neural network can take it as an input (4 dimensions)

# In[ ]:


X_train = X_train.values.reshape(-1,28,28,1)


# In[ ]:


test_data = test.reshape(-1, 28, 28, 1)


# Keras uses one hot encoding
# 
# Hence changing the labels from integer to categorical data

# In[ ]:


train_labels_one_hot = to_categorical(train_lbls)

#Display the category label using one-hot encoding
print('Original label : ', train_lbls[2])
print('After conversion to categorical ( one-hot ) : ', train_labels_one_hot[2])


# **Our data is now ready to be inputted into a neural network**

# ## Making the Keras Model

# In[ ]:


model=Sequential()

model.add(Conv2D(kernel_size=5,strides=1,filters=64,
                padding='same',activation='relu', input_shape = (28,28,1)))
model.add(Conv2D(kernel_size=5,strides=1,filters=64,
                padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=2,strides=2))
model.add(Dropout(0.25))

model.add(Conv2D(kernel_size=5,strides=1,filters=64,
                padding='same',activation='relu'))
model.add(Conv2D(kernel_size=5,strides=1,filters=64,
                padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=2,strides=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))


# Adding dropout layers to reduce overfitting
# 
# [Read more about layers here](https://forums.fast.ai/t/dense-vs-convolutional-vs-fully-connected-layers/191/2 "Layers info.")
# 
# The last Dense layer has 10 units (since we need to classify into 10 classes (0-9))

# In[ ]:


model.summary()


# Using the Adam optimizer

# In[ ]:


from keras.optimizers import Adam
optimizer=Adam(lr=1e-3)


# ### Compiling the model, using the Adam optimizer, with a categorical crossentropy loss so we can predict into multiple classes

# In[ ]:


model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=["accuracy"])


# ### Fitting out data onto the model, with 256 images at a time, for 10 epochs.

# In[ ]:


history = model.fit(x=X_train, y=train_labels_one_hot,
                    batch_size=256, epochs=10, verbose=1)


# ### Plotting the accuracy

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
#Checking for overfitting

#Plot the Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Plot',fontsize=16)


# ### Predicting the values of the test data

# In[ ]:


predictions = model.predict(test_data)

predictions = [np.argmax(i) for i in predictions]
len(predictions)


# ### Plotting some image predictions from our neural model

# In[ ]:


img_shape=(28,28)

def plot_images(images, cls_pred=None):
    assert len(images) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        xlabel = "Pred: {0}".format(cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# In[ ]:


images = test_data[9:18]

y_pred=model.predict(x=images)
cls_pred=np.argmax(y_pred,axis=1)

cls_pred


# In[ ]:


y_pred = model.predict(x=images)
cls_pred = np.argmax(y_pred, axis=1)


# In[ ]:


plot_images(images=images,
           cls_pred=cls_pred)


# ### Hence we have used Neural Networks to successfully predict handwritten image classes, with pretty high accuracy

# ### Making a submission file for Kaggle:

# In[ ]:


data = list(zip(list(range(1, 28001)), predictions))

submission_df = pd.DataFrame(data, columns=['ImageId','Label'])
submission_df.head()


# In[ ]:


submission_df.to_csv("Submission.csv",index=False)

