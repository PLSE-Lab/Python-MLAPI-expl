#!/usr/bin/env python
# coding: utf-8

# #  import libraries

# In[ ]:


import cv2                 
import numpy as np         
import os                  
from random import shuffle
import tensorflow as tf 
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob as gb
from tensorflow.keras.utils import to_categorical


# In[ ]:


#read DataSet
TrianImage="/kaggle/input/chest-xray-covid19-pneumonia/Data/train/"
TestImage="/kaggle/input/chest-xray-covid19-pneumonia/Data/test/"


print(TrianImage)
print(TestImage)


# In[ ]:


#to get all image names in train file
Pneumonaimages = os.listdir(TrianImage + "/PNEUMONIA")
Normalimages = os.listdir(TrianImage + "/NORMAL")
COVID19images = os.listdir(TrianImage + "/COVID19")





# # # **Explore the Data**

# In[ ]:


#plot to show the size of some image
#plot PNEUMONIA
plt.figure(figsize=(20,10))
for i in range(6):
    plt.subplot(3, 3, i + 1)
    plt.imshow(plt.imread(os.path.join(TrianImage + "/PNEUMONIA",Pneumonaimages[i])),cmap='gray')
    plt.title("PNEUMONIA")
    
plt.show()
#plot NORMAL
plt.figure(figsize=(20,10))
for i in range(6):
    plt.subplot(3, 3, i + 1)
    plt.imshow(plt.imread(os.path.join(TrianImage + "/NORMAL",Normalimages[i])),cmap='gray')
    plt.title("NORMAL")

plt.show()
#plot 
plt.figure(figsize=(20,10))
for i in range(6):
    plt.subplot(3, 3, i + 1)
    plt.imshow(plt.imread(os.path.join(TrianImage + "/COVID19",COVID19images[i])),cmap='gray')
    plt.title("COVID19")


# # ImageDataGenerator (DataAugmentation )

# We also use the generator to transform the values in each batch so that their mean is $0$ and their standard deviation is 1.
# This will facilitate model training by standardizing the input distribution

# The generator also converts our single channel X-ray images (gray-scale) to a three-channel format by repeating the values in the image across all channels.
# We will want this because the pre-trained model that we'll use requires three-channel inputs.

# In[ ]:


train_datagen = ImageDataGenerator(
      samplewise_center=True,
      samplewise_std_normalization= True,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      fill_mode='nearest'
                                  )

# NOTE: YOU MUST USE A BATCH SIZE OF 10 (batch_size=10) FOR THE 
# TRAIN GENERATOR.
train_generator =train_datagen.flow_from_directory(
     TrianImage,
     batch_size= 256,
     shuffle=shuffle,
     target_size=(300, 300)

)

test_generator =train_datagen.flow_from_directory(
     TestImage,
     batch_size= 50,
     shuffle=shuffle,
     target_size=(300, 300)

)


# In[ ]:


trainShape=train_generator.__getitem__(0)[0].shape
testShape=test_generator.__getitem__(0)[0].shape


# In[ ]:


#show labels
test_generator.__getitem__(0)[1]


# In[ ]:


#Shape of Data
print("Train Shape \n",trainShape)
print("Test Shape \n",testShape)


# In[ ]:




Labels={'COVID19':0,'NORMAL':1 ,'PNEUMONIA':2 }

# convert label to code
def getCode(label):
    return Labels[label]


# convert code to label 
def getLabel(n):
    for x,c in Labels.items():
        if n==c:
            return x
        
        
        
#Test        
print(getCode('COVID19'))
print(getLabel(1))


# # Explore Data After DataAugmentation and standardizing 

# In[ ]:


plt.figure(figsize=(20,10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(train_generator.__getitem__(0)[0][i])
    plt.title(getLabel(np.argmax(train_generator.__getitem__(0)[1][i])))


# # Another Way To Load Data

# In[ ]:


#Reading image data
import glob as gb
import cv2  
sizeImage=300 # to resize the all image as same size

#to read all images from directory
def getData(Dir,sizeImage):
    X=[]
    y=[]
    for folder in  os.listdir(Dir) : #to get the file name 
        files = gb.glob(pathname= str( Dir  +"/" +folder+ '//*.jpg' )) # to get the images
        for file in files:
                picture=cv2.imread(file) #  or plt.imread(file)
                imageArray=cv2.resize(picture,(sizeImage,sizeImage))
                X.append(list(imageArray))
                y.append(getCode(folder))
    X=np.array(X)
    y=np.array(y)
    return X,y



# In[ ]:


#get train data
X_train, y_train = getData(TrianImage,sizeImage)
# get test data
X_test , y_test = getData(TestImage,sizeImage)


# In[ ]:


print("X_train Shape        ",X_train.shape)
print("X_test Shape         ",X_test.shape)


# In[ ]:


# #Convert y_train to categorical
y_train=to_categorical(y_train,3)
print("y_train ",y_train.shape)



#Convert y_train to categorical
y_test=to_categorical(y_test,3)
print("y_test ",y_test.shape)


# # Build Model

# In[ ]:


#load weight
Network_Weight="/kaggle/input/densenet-keras/DenseNet-BC-169-32-no-top.h5"
print(Network_Weight)


# In[ ]:



from tensorflow.keras.applications.densenet import DenseNet169
pre_trained_model = DenseNet169(input_shape = (sizeImage, sizeImage, 3), 
                               include_top = False, 
                               weights = None)
pre_trained_model.load_weights(Network_Weight)
for layer in pre_trained_model.layers:
   layer.trainable = False  #to make the layers to Freeze Weights
pre_trained_model.summary()


# In[ ]:


from tensorflow.keras import Model


x = tf.keras.layers.Flatten()(pre_trained_model.output)

#Full Connected Layers
x = tf.keras.layers.Dense(512, activation='relu')(x)
#Add dropout to avoid Overfit
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
#Add dropout to avoid Overfit
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)


x=tf.keras.layers.Dense(3 , activation='sigmoid')(x)
       

model = Model( pre_trained_model.input, x) 

print(model.summary())
model.compile(optimizer='adam', loss="binary_crossentropy",metrics=['accuracy'])


# In[ ]:


from tensorflow.keras.callbacks import ReduceLROnPlateau , ModelCheckpoint
lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=1, verbose=1)

filepath="transferlearning_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


# In[ ]:


epochs = 12
history = model.fit_generator(train_generator,steps_per_epoch=20,callbacks=[lr_reduce,checkpoint] ,
         epochs=epochs)


# In[ ]:


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


# In[ ]:





# In[ ]:


#Evaluate Model
model.evaluate(test_generator)


# In[ ]:


model.save('modelCovid19___4.h5')


# In[ ]:


#prediction
pred=model.predict(test_generator)


# In[ ]:


print(len(pred))


# In[ ]:


y_test=[]
for i in range(26):
    y_test.extend(test_generator.__getitem__(i)[1])


# In[ ]:


print(len(y_test))
y_test=np.array(y_test)


# In[ ]:


y_test=np.argmax(y_test,axis=1)
pred= np.argmax(pred,axis=1)


# In[ ]:


print("pred \n",len(pred))
print("y_test \n",len(y_test))


# In[ ]:


print("y_test \n",y_test)
print("pred \n",pred)


# In[ ]:


#confusion_matrix to check in accuracy 
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(pred,y_test)
print(cm)


# In[ ]:


plt.figure(figsize=(20,10))
for i in range(0,9):
    
    plt.subplot(3, 3, i + 1)
    
    plt.imshow(test_generator.__getitem__(0)[0][i],cmap='gray')
    plt.title(f"   Real: {getLabel(y_test[i])   } Vs  Predict: {getLabel(pred[i])}")


# In[ ]:


#lto load model
from keras.models import load_model
loadedModel=load_model("modelCovid19___2.h5")


# In[ ]:


loadedModel.compile(optimizer='adam', loss="binary_crossentropy",metrics=['accuracy'])
loadedModel.evaluate(test_generator)

