#!/usr/bin/env python
# coding: utf-8

# 
# # Getting started with Keras | CNN implementation for Digit Recognizer 

# ## Content 
# ---
# - ### Loading the Dataset
#         - Splitting labels from images
#         - Visulaizing an image on pixel level
#         - Visulaizing sample of the Training data 
#         - Visualizing sample of the Testing Data 
#  
# - ### Data Preparation
#         - Reshape images in matrix form
#         - Normalization
#         - Applying One-hot-encoding on labels
# 
# - ### Network structure
#         - Model generator to dynamically change hyperparameters
# 
# - ### Training
#         - Training generator and fitting
#         - Visualize Model behaviour on loss drawing
# 
# - ### Evaluation
#         - Predict classes for testing data
#         - Visulaize predictions on images
#         - Saving results
#       
# 
# ## History
# ---
# Using Dense Nodes best results was **0.97571**
# 
# ---
# 

# In[ ]:


# Importing Modules
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))


# ---
# # Loading Dataset

# In[ ]:


# Reading the data frames
#Training Data
train = pd.read_csv("../input/train.csv") 
#Testing Data
test = pd.read_csv("../input/test.csv")


# # Splitting Labels from the images

# In[ ]:


#Converting the Pandas frame to numpy 
label = np.array(train.iloc[:,0],np.str)
data = np.array(train.iloc[:,1:],np.float32)

#Test data
label_test = np.array([])
data_test = np.array(test.iloc[:,:],np.float32)


# # Visulaizing an Image on the pixel level

# In[ ]:


def visualize_input(img, ax):
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y],2)), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y]<thresh else 'black')

fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
visualize_input(data[4].reshape(28,28), ax)


# # Visualizing sample of the training dataset

# In[ ]:


#Drawing Train Data
fig = plt.figure(figsize=(20,20))
for i in range(10):
    ax = fig.add_subplot(1,10,i+1)
    ax.imshow(np.reshape(data[i],(28,28)),cmap='gray')
    ax.set_title(str(label[i]))


# # Visualizing sample of the testing dataset

# In[ ]:


#Drawing Test Data 
fig = plt.figure(figsize=(20,20))
for i in range(10):
    ax = fig.add_subplot(1,10,i+1)
    ax.imshow(np.reshape(data_test[i],(28,28)),cmap='gray')


# ---
# # Data Preparation
# 
# - The dataset is clean no need for cleaning
# - Reshap the data to  (height = 28px, width = 28px , canal = 1) for 1D Vector
# - Normalizing the data as each pixel is from 0 : 255 => Black => white in gray scale
# - the data will be from 0 to 1

# In[ ]:


# Reshape and Normalizing the data
#(height = 28px, width = 28px , canal = 1)
data = data.reshape(data.shape[0],28,28,1)
data_test = data_test.reshape(data_test.shape[0],28,28,1)
data = data/255
data_test = data_test/255


# In[ ]:


data.shape


# ### One-hot-encoding | Convering the label to one hot encoded 10 categories each for one number 

# In[ ]:


from keras.utils import np_utils
print("Before conding")
print(label[:10])
labels = np_utils.to_categorical(label,10)
print("Encoded Data")
print(labels[:10])


# --- 
# ## Network Structure
# Creating Model generators to dynamically change and select the best hyperparameters for the model  using the same structure.
# 
# - Dropout parameter
# - Denses 
# - Activation layer

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Activation, Dropout

def model_generator(dropout=[0.25],denses=[512,10],activation="relu"):
    model = Sequential()
    model.add(Conv2D(filters=32,kernel_size=3,padding='same', activation='relu', input_shape=(28, 28,1)))
    model.add(Conv2D(filters=32, kernel_size=3,  border_mode='same', activation='relu'))
    model.add(MaxPool2D(pool_size=3))
    model.add(Dropout(0.20))
    
    model.add(Conv2D(filters=64,kernel_size=3,padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3,  border_mode='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters=128, kernel_size=3,  border_mode='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(0.20))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))  
    model.add(Dense(10))
    model.add(Activation('softmax'))
    #Model Summary
    model.summary()
    return model


# In[ ]:


def model_generator2(dropout=[0.25],denses=[512,10],activation="relu"):
    model = Sequential()
    model.add(Conv2D(filters=16,kernel_size=2,padding='same', activation='relu', input_shape=(28, 28,1)))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(0.20))
    model.add(Conv2D(filters=32, kernel_size=2,  border_mode='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(0.20))
    model.add(Conv2D(filters=64,kernel_size=2,padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(0.15))
    #model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(512, name='aux_output'))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, name='aux_output2'))
    model.add(Activation('softmax'))
    #Model Summary
    model.summary()
    return model


# ---
# # Creating Training Generator 
# Using keras callback to save the best model based on **validation accuracy** through the training to prevent overfitting with the validation dataset

# In[ ]:


from keras.optimizers import RMSprop, Adam
def model_fit(model,batch_size=64,epochs=10):
#     optimizer = RMSprop(lr=0.0002, rho=0.9, epsilon=1e-08, decay=0.0)
    optimizer = Adam(lr=0.0001)
    model.compile(loss="categorical_crossentropy",optimizer=optimizer,metrics=['accuracy'])
    from keras.callbacks import ModelCheckpoint
    checkpointer = ModelCheckpoint(filepath='mnist.model.best', verbose=1, monitor='val_loss', save_best_only=True)
    training = model.fit(data, labels,batch_size=batch_size, epochs=epochs,validation_split=0.25, callbacks=[checkpointer],verbose=1, shuffle=True)
    return training


# # Start Training

# In[ ]:


model1 = model_generator(dropout=[0.25],denses=[128,10],activation="relu")
training = model_fit(model1,batch_size=128,epochs=100)


# # Visualize model behaviour

# In[ ]:


def draw_model(training):
    plt.plot(training.history['loss'])
    plt.plot(training.history['val_loss'],'r')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend(["Training","Validation"])
    plt.show()
    plt.plot(training.history['acc'])
    plt.plot(training.history['val_acc'])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Accuracy")
    plt.legend(["Training","Validation"],loc=4)
    plt.show()
    
draw_model(training)


# ---
# # Model Scoring on Test set

# In[ ]:


# evaluate test accuracy
def scoring(model):
    model.load_weights('mnist.model.best')
    score = model.evaluate(data[:2000], labels[:2000], verbose=0)
    accuracy = 100*score[1]
    # print test accuracy
    print('Test accuracy: %.4f%%' % accuracy)
    label_test = model.predict_classes(data_test)
    print("Sample of the prdiction",label_test[:10])
    return label_test


# In[ ]:


label_test = scoring(model1)


# # Drawing predicited data 

# In[ ]:


#Drawing Test Dta 
fig = plt.figure(figsize=(20,20))
for i in range(10):
    ax = fig.add_subplot(1,10,i+1)
    ax.imshow(np.reshape(data_test[i],(28,28)),cmap='gray')
    ax.set_title(label_test[i])


# In[ ]:


#Drawing Test Dta 
fig = plt.figure(figsize=(20,20))
for i in range(10):
    rn = np.random.randint(1,100)
    ax = fig.add_subplot(1,10,i+1)
    ax.imshow(np.reshape(data_test[rn],(28,28)),cmap='gray')
    ax.set_title(label_test[rn])


# # Saving the scores to the submission CSV

# In[ ]:


np.savetxt("submission.csv", np.dstack((np.arange(1, label_test.size+1),label_test))[0],"%d,%d",header="ImageId,Label",comments="")


# In[ ]:





# In[ ]:




