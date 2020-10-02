#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import os
import numpy as np
import pandas as pd
from tensorflow import keras

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# # Why reshape required??
# Each image (instance) in the dataset has 784 pixels (features) and value of each feature(pixel) ranges from 0 to 255, this range is too wide, hence we have performed Normalization on the training and test dataset, by dividing the pixels by 255, so that values of all features (pixels) are in a small range (0 to 1).

# In[ ]:


class LoadData():
    def __init__(self):
        self.data_train = pd.read_csv(os.path.join('..','input','fashionmnist',"fashion-mnist_train.csv"))
        self.data_test = pd.read_csv(os.path.join('..','input','fashionmnist',"fashion-mnist_test.csv"))
        
        self.x_train = np.array(self.data_train.iloc[:,1:])
        self.y_train = np.array(self.data_train.iloc[:,0])
        
        self.x_test = np.array(self.data_test.iloc[:,1:])
        self.y_test = np.array(self.data_test.iloc[:,0])
        
    def reshape_data(self):
        img_rows, img_cols = 28, 28
        print("Before: ")
        print(self.x_train.shape,self.y_train.shape)
        print(self.x_test.shape,self.y_test.shape)
        
        self.x_train = self.x_train.reshape(self.x_train.shape[0],img_rows,img_cols,1).astype('float32')
        self.x_test = self.x_test.reshape(self.x_test.shape[0],img_rows,img_cols,1).astype('float32')
        self.x_train /= 255
        self.x_test /= 255
        
        print("After: ")
        
        
        self.y_train = keras.utils.to_categorical(self.y_train)
        self.y_test = keras.utils.to_categorical(self.y_test)
        self.num_classes = self.y_test.shape[1]
        print(self.x_train.shape,self.y_train.shape)
        print(self.x_test.shape,self.y_test.shape)
        print("total classes: ",self.num_classes)


# In[ ]:


ld_obj = LoadData()
ld_obj.reshape_data()


# In[ ]:


class DesignModel():
    def __init__(self):
        self.model = None
        self.history = None
        self.x_train = ld_obj.x_train
        self.y_train = ld_obj.y_train
        
        self.x_test = ld_obj.x_test
        self.y_test = ld_obj.y_test
        self.num_classes = ld_obj.num_classes
        print(self.num_classes)


    def create_model(self):
        self.model = keras.Sequential()
        self.model.add(keras.layers.Conv2D(32, (3,3), activation='relu',input_shape=(28,28,1)))
        #self.model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
        #self.model.add(keras.layers.Conv2D(128, (3,3), activation='relu'))
        self.model.add(keras.layers.MaxPool2D((2,2)))
        self.model.add(keras.layers.Flatten())
        #self.model.add(keras.layers.Dense(128, activation='relu'))
        self.model.add(keras.layers.Dense(self.num_classes, activation='softmax'))
        print(self.model.summary())
    
    def compile_model(self):
        self.model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
    
    def train_model(self,batch_size,epochs):
        self.history = self.model.fit(self.x_train,self.y_train,epochs=epochs,
                                      batch_size=batch_size,verbose=2,
                                      validation_data = (self.x_test,self.y_test),shuffle=True)


# In[ ]:


epochs = 20
batch_size = 512

model_obj = DesignModel()
model_obj.create_model()
model_obj.compile_model()
model_obj.train_model(batch_size,epochs)


# In[ ]:


class EvaluateModel():
    def __init__(self,history):
        self.history = history
        
    def plot_acc_loss(self):
        epochs = self.history.epoch
        acc = self.history.history.get('accuracy')
        loss = self.history.history.get('loss')

        print(epochs,loss,acc)
        plt.plot(epochs, acc, 'b', label='Training accuracy')
        plt.title('Training accuracy')

        plt.figure()

        plt.plot(epochs, loss, 'b', label='Training Loss')
        plt.title('Training loss')
        plt.legend()

        plt.show()
        


# In[ ]:


eval_obj = EvaluateModel(model_obj.history)
eval_obj.plot_acc_loss()


# In[ ]:


from PIL import Image
from IPython.display import Image as IMG


class Prediction():
    
    def __init__(self,model):
        self.model = model
        labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.labels = {k:labels[k] for k in range(len(labels))}
    
    def predict(self,image):
        img = Image.open(image)
        print("Image: ")
        display(IMG(filename=image))
        img = img.convert("L")
        img = img.resize((28,28))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,28,28,1)
        y_pred = self.model.predict(im2arr)
        return y_pred
        
pred_obj = Prediction(model_obj.model)
files = None

for r,d,f in os.walk(os.path.join(cur_dir,"Test_Data")):
    files = f
for item in files:
    result = pred_obj.predict(os.path.join(cur_dir,"Test_Data",item))
    print("Predition Label: ",pred_obj.labels[result.argmax()],'\n')


# In[ ]:




