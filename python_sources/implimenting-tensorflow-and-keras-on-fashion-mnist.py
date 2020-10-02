#!/usr/bin/env python
# coding: utf-8

# ## Data Description
# Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255. The training and test data sets have 785 columns. The first column consists of the class labels (see above), and represents the article of clothing. The rest of the columns contain the pixel-values of the associated image.
# ### Labels
# Each training and test example is assigned to one of the following labels:
# 
# 0 T-shirt/top
# 
# 1 Trouser
# 
# 2 Pullover
# 
# 3 Dress
# 
# 4 Coat
# 
# 5 Sandal
# 
# 6 Shirt
# 
# 7 Sneaker
# 
# 8 Bag
# 
# 9 Ankle boot
# 
# 
# 
# Each row is a separate image
# Column 1 is the class label.
# Remaining columns are pixel numbers (784 total).
# Each value is the darkness of the pixel (1 to 255)

# ### Loading useful libraries

# In[ ]:


# Input      data     files     are    available in the "../input/" directory.

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Import Pandas for data manipulation using dataframes

import pandas as pd

#Import Numpy for statistical calculations

import numpy as np

# Import Warnings 

import warnings

warnings.filterwarnings('ignore')

# Import matplotlib Library for data visualisation

import matplotlib.pyplot as plt

#Import train_test_split from scikit library

from sklearn.model_selection import train_test_split

# Import Keras

import keras

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

from keras.optimizers import Adam

from keras.callbacks import TensorBoard

num_classes = 10

epochs = 5


# In[ ]:


#Loading test and train data
train_df = pd.read_csv('/kaggle/input/fashion-mnist_train.csv',sep=',')
test_df = pd.read_csv('/kaggle/input/fashion-mnist_test.csv', sep = ',')


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# Now let us split the train data into x and y arrays where x represents the image data and y represents the labels.
# 
# To do that we need to convert the dataframes into numpy arrays of float32 type which is the acceptable form for tensorflow and keras.
# 

# In[ ]:



train_data = np.array(train_df, dtype = 'float32')


# In[ ]:


#Similarly let us do the same process for test data

test_data = np.array(test_df, dtype='float32')


# Now let us slice the train arrays into x and y arrays namely x_train,y_train to store all image data and label data respectively. i.e
# 
#     x_train contains all the rows and all columns except the label column and excluding header info .
# 
#     y_train contains all the rows and first column and excluding header info .
# 
# Similarly slice the test arrays into x and y arrays namely x_test,y_test to store all image data and label data respectively. i.e
# 
#     x_test contains all the rows and all columns except the label column and excluding header info .
#     y_test contains all the rows and first column and excluding header info .
# 

# In[ ]:


train_df.describe()


# from above : Since the image data in x_train and x_test is from 0 to 255 , we need to rescale this from 0 to 1.To do this we need to divide the x_train and x_test by 255
# 

# In[ ]:


x_train = train_data[:,1:]/255

y_train = train_data[:,0]

x_test= test_data[:,1:]/255

y_test=test_data[:,0]


# Now split the training data into validation and actual training data for training the model and testing it using the validation set. This is achieved using the train_test_split method of scikit learn library.
# 
# 

# In[ ]:


x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size = 0.2,random_state = 12345)


# In[ ]:


#Now let us visualise the sample image how it looks like in 28 * 28 pixel size

image = x_train[55,:].reshape((28,28))
plt.imshow(image)
plt.show()


# ## Creating convolutional Neural Networks
#     1.Define the model
# 
#     2.Compile the model
#     
#     3.Fit the model

# In[ ]:


# Defined the shape of the image as 3d with rows and columns and 1 for the 3d visualisation

image_rows = 28

image_cols = 28

batch_size = 512

image_shape = (image_rows,image_cols,1) 


# In[ ]:


#formating on the x_train,x_test and x_validate sets.

x_train = x_train.reshape(x_train.shape[0],*image_shape)
x_test = x_test.reshape(x_test.shape[0],*image_shape)
x_validate = x_validate.reshape(x_validate.shape[0],*image_shape)


# ### Defininig the model

# In[ ]:


cnn_model = Sequential([
    Conv2D(filters=32,kernel_size=3,activation='relu',input_shape = image_shape),
    MaxPooling2D(pool_size=2) ,# down sampling the output instead of 28*28 it is 14*14
    Dropout(0.2),
    Flatten(), # flatten out the layers
    Dense(32,activation='relu'),
    Dense(10,activation = 'softmax')
    
])


# ### Compiling model

# In[ ]:


cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])


# ### fit the model

# In[ ]:


history = cnn_model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=50,
    verbose=1,
    validation_data=(x_validate,y_validate),
)


# ### Evaluate / score of the model

# In[ ]:


score = cnn_model.evaluate(x_test,y_test,verbose=0)
print('Test Loss : {:.4f}'.format(score[0]))
print('Test Accuracy : {:.4f}'.format(score[1]))


# ### plotting the result ie training accuracy vs validation accuracy and training loss and validation loss

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# 
# ### summarizing the performance of our classifier as follows
# 
# 

# In[ ]:


#get the predictions for the test data
predicted_classes = cnn_model.predict_classes(x_test)
#get the indices to be plotted
y_true = test_df.iloc[:, 0]
correct = np.nonzero(predicted_classes==y_true)[0]
incorrect = np.nonzero(predicted_classes!=y_true)[0]
from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_true, predicted_classes, target_names=target_names))


# It's apparent that our classifier is underperforming for class 6 in terms of both precision and recall. For class 2, classifier is slightly lacking precision whereas it is slightly lacking recall (i.e. missed) for class 4.
# 
# 
# Perhaps we would gain more insight after visualizing the correct and incorrect predictions.
# 

# 
# ### Subset of correctly predicted classes.

# In[ ]:


for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_true[correct]))
    plt.tight_layout()


# ### Subset of incorrectly predicted classes.
# 

# In[ ]:



for i, incorrect in enumerate(incorrect[0:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_true[incorrect]))
    plt.tight_layout()


# ### Predict and Submit results 

# In[ ]:




# predict results
 
results = cnn_model.predict(x_test)

# select the indix with the maximum probability
 
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,10001),name = "id"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)


# In[ ]:





# In[ ]:





# In[ ]:




