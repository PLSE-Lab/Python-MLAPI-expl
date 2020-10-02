#!/usr/bin/env python
# coding: utf-8

# # Malaria Cell CNN Classifier
# 
# The data of this project comes from Kaggle.com(https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria).
# 
# With data of Malaria Cell, we can build a CNN model and train the model with image given to tell whether a cell is infected or not.

# In[ ]:


import os
import sys
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.preprocessing import image
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from IPython.display import SVG
from keras.layers import Dense, Conv2D, MaxPooling2D,Dropout,BatchNormalization,Flatten


# First we load images in two folders.The dataset is well set with same size in two categories.

# In[ ]:


parapath = '../input/cell_images/cell_images/Parasitized/'
uninpath = '../input/cell_images/cell_images/Uninfected/'
parastized = os.listdir(parapath)
uninfected = os.listdir(uninpath)
print(sys.getsizeof(parastized))
print(sys.getsizeof(parastized))


# Load all the image the path collected in prastized and uninfected list. In Windows there is a image thumbnail cache "Thumb.db" so we dont load that in our list.
# 
# Change them into arrays and save in list of data.To make a target array we set up a label list with 1 to a parastized cell and 0 to an uninfected one.

# In[ ]:


data = []
label = []
for para in parastized:
    try:
        img =  image.load_img(parapath+para,target_size=(112,112))
        x = image.img_to_array(img)
        data.append(x)
        label.append(1)
    except:
        print("Can't add "+para+" in the dataset")
for unin in uninfected:
    try:
        img =  image.load_img(uninpath+unin,target_size=(112,112))
        x = image.img_to_array(img)
        data.append(x)
        label.append(0)
    except:
        print("Can't add "+unin+" in the dataset")


# Change list data to array in Numpy. Note that the variable data now has a memory of 5 GB. 
# 
# I tried to use (224,224,3)rather than (112,112,3) as input of VGG-16, but found variable data will have a size of 15GB and raise a MemoryError.

# In[ ]:


data = np.array(data)/255
label = np.array(label)


# In[ ]:


print(sys.getsizeof(data))
print(data.shape)


# Now we normlize the data matrix, the range of a PGB pixel is (0,255) so divide by 255 is OK to let values in range of (0,1).
# 
# Shuffle and Split data to train and test with sklearn.model_selection.train_test_split()

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(data,label,test_size = 0.1,random_state=0)


# Keras is a powerful python library to set up neural network models. The model set is similiar to VGG with fewer parameters and layers. 

# In[ ]:


def MalariaModel():
    model = Sequential()
    model.add(Conv2D(filters = 4, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'a11', input_shape = (112, 112, 3)))  
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'a12'))
    model.add(BatchNormalization(name = 'a13'))
    #input = (112,112,4)
    model.add(Conv2D(filters = 8, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'a21'))   
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'a22'))
    model.add(BatchNormalization(name = 'a23'))
    #input = (56,56,8)
    model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'a31'))   
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'a32'))
    model.add(BatchNormalization(name = 'a33'))
    #input = (28,28,16)
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'a41'))   
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'a42'))
    model.add(BatchNormalization(name = 'a43'))
    #input = (14,14,32)
    model.add(Flatten())
    model.add(Dense(32, activation = 'relu', name = 'fc1'))
    model.add(Dense(1, activation = 'sigmoid', name = 'prediction'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# The model has about 50000 parameters

# In[ ]:


modelone = MalariaModel()
modelone.summary()


# Train the model with training set. 
# 
# The model fits training data just in one epoch and I found any additional epochs will make prediction overfit and become a random guess.

# In[ ]:


output = modelone.fit(x_train, y_train,validation_split=0.1,epochs=4, batch_size=50)


# The model receives an accuracy about 95% on test set.

# In[ ]:


preds = modelone.evaluate(x = x_test,y = y_test)
print("Test Accuracy : %.2f%%" % (preds[1]*100))


# In[ ]:


plt.plot(output.history['acc'])
plt.plot(output.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'cross-validation'], loc='upper left')
plt.savefig('Accuracy.png',dpi=100)
plt.show()


# Here is a figure of model's architecture

# In[ ]:


modelone.save('malariaCNNModel.h5')


# In[ ]:


modelpic = plot_model(modelone, to_file='model.png')
SVG(model_to_dot(modelone).create(prog='dot', format='svg'))


# In[ ]:




