#!/usr/bin/env python
# coding: utf-8

# ## A Fast and Easy CNN Kernel in Keras for Beginners - 0.99471 accuracy   
# 
# The aim of this kernel is to provide beginners (including me) a fast and easy platform to build a Convolutional Neural Network (CNN) model. This  notebook is focused on the coding side. For detailed description, please visit my web site at http://www.codeastar.com/convolutional-neural-network-python/ .

# First thing first, let's load training and testing data sets from Kaggle. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# Then import required modules. 

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Have a quick look on our training data.

# In[ ]:


df_train.head()


# We load the 2nd to 785th columns, the pixel value columns, as our input X.
# And load the first label column as output Y. 

# In[ ]:


#every columns but the first
df_train_x = df_train.iloc[:,1:] 
#only the first column
df_train_y = df_train.iloc[:,:1] 


# We can use *imshow()* from *matplotlab* to display the our input X as images. See rather those images can map with the values of output Y. 

# In[ ]:


ax = plt.subplots(1,5)
for i in range(0,5):   #validate the first 5 records
    ax[1][i].imshow(df_train_x.values[i].reshape(28,28), cmap='gray')
    ax[1][i].set_title(df_train_y.values[i])


# ### It's building time. 
# We use Keras to build our CNN model, with typical LeNet architecture (http://yann.lecun.com/exdb/lenet/).  But unlike other popular Keras kernels on Kaggle, I do not apply the stacking convolutional layers here. As I find this is just unnecssary to spend that much resources on simple digit image sets. 

# In[ ]:


def cnn_model(result_class_size):
    model = Sequential()
    #use Conv2D to create our first convolutional layer, with 32 filters, 5x5 filter size, 
    #input_shape = input image with (height, width, channels), activate ReLU to turn negative to zero
    model.add(Conv2D(32, (5, 5), input_shape=(28,28,1), activation='relu'))
    #add a pooling layer for down sampling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # add another conv layer with 16 filters, 3x3 filter size, 
    model.add(Conv2D(16, (3, 3), activation='relu'))
    #set 20% of the layer's activation to zero, to void overfit
    model.add(Dropout(0.2))
    #convert a 2D matrix in a vector
    model.add(Flatten())
    #add fully-connected layers, and ReLU activation
    model.add(Dense(130, activation='relu'))
    model.add(Dense(50, activation='relu'))
    #add a fully-connected layer with softmax function to squash values to 0...1 
    model.add(Dense(result_class_size, activation='softmax'))   
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model


# Model is done, let's take a look on our model summary. 

# In[ ]:


#turn the label to 42000 binary class matrix 
arr_train_y = np_utils.to_categorical(df_train_y['label'].values)
model = cnn_model(arr_train_y.shape[1])
model.summary()


# When there is more trainable params, more time is needed. And in the world of machine learning and optimization, more parameters may not always provide better result, as it may cause an overfitting issue. Anyway, we are good at current status.

# In[ ]:


#normalize 255 grey scale to values between 0 and 1 
df_test = df_test / 255
df_train_x = df_train_x / 255


# Since our model uses 28x28 pixel matrix as input, we then need to reshape both our training and testing input Xs. 

# In[ ]:


#reshape training X and text x to (number, height, width, channels)
arr_train_x_28x28 = np.reshape(df_train_x.values, (df_train_x.values.shape[0], 28, 28, 1))
arr_test_x_28x28 = np.reshape(df_test.values, (df_test.values.shape[0], 28, 28, 1))


# In[ ]:


random_seed = 3
#validate size = 8%
split_train_x, split_val_x, split_train_y, split_val_y, = train_test_split(arr_train_x_28x28, arr_train_y, test_size = 0.08, random_state=random_seed)


# A reduce learinging callback is prepared for later use. When there is no improvement in our model after **3** training rounds, the new learining rate will be calculated as "current learning rate x factor( **0.5** )" . 

# In[ ]:


reduce_lr = ReduceLROnPlateau(monitor='val_acc', 
                              factor=0.5,
                              patience=3, 
                              min_lr=0.00001)


# We also apply the image generator function to cover more  variance of data distribution. 

# In[ ]:


datagen = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range 
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally
        height_shift_range=0.1  # randomly shift images vertically
        )

datagen.fit(split_train_x)


# After we add the image generator and learning rate callback to model, we can start training it in 30 epochs.

# In[ ]:


model.fit_generator(datagen.flow(split_train_x,split_train_y, batch_size=64),
                              epochs = 30, validation_data = (split_val_x,split_val_y),
                              verbose = 2, steps_per_epoch=700 
                              , callbacks=[reduce_lr])


# It needs around 1200 seconds (20 minutes) to run the training process. 

# In[ ]:


prediction = model.predict_classes(arr_test_x_28x28, verbose=0)
data_to_submit = pd.DataFrame({"ImageId": list(range(1,len(prediction)+1)), "Label": prediction})
data_to_submit.to_csv("result.csv", header=True, index = False)


# After a cup of coffee (a tea for myself, as I don't drink coffee usually), we can get the prediction, save it as a csv file and submit to Kaggle. But before sending out our submission, let' see how well our trained model is, by our own eyes.

# In[ ]:


from random import randrange
#pick 10 images from testing data set
start_idx = randrange(df_test.shape[0]-10) 


# In[ ]:


fig, ax = plt.subplots(2,5, figsize=(15,8))
for j in range(0,2): 
  for i in range(0,5):
     ax[j][i].imshow(df_test.values[start_idx].reshape(28,28), cmap='gray')
     ax[j][i].set_title("Index:{} \nPrediction:{}".format(start_idx, prediction[start_idx]))
     start_idx +=1


# Satisfied with the result? The model is relatively fast and simple and gives me a good 0.99471 accuracy. Feel free to tweak / enhance it to get your upgraded version.  
