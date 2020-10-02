#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import cv2
from keras.datasets import fashion_mnist#download mnist data and split into train and test sets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout#create model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import VGG16


# ## STAGE 1 = GENERATING NUMPY DATASETS AND SAVING THEM
# ## STAGE 2 = USING THE NUMPY DATASETS TO RUN MODELS AND SAVING WEIGHTS
# ## STAGE 3 = LOADING WEIGHTS OF ABOVE MODEL AND USING THEM
# ## STAGE 4 = TRYING DATA AUGMENTATION BY USING DATASETS FROM STAGE 1
# ## STAGE 5 = TRYING TRANSFER LEARNING

# # <div align="center">STAGE I</div>

# In[ ]:


CAT_TRAIN_PATH="/kaggle/input/cat-and-dog/training_set/training_set/cats/"
DOG_TRAIN_PATH="/kaggle/input/cat-and-dog/training_set/training_set/dogs/"
CAT_TEST_PATH="/kaggle/input/cat-and-dog/test_set/test_set/cats/"
DOG_TEST_PATH="/kaggle/input/cat-and-dog/test_set/test_set/dogs/"


# In[ ]:


def preprocess_img(img):
    dim=(100,100)
    res = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
    return res
    


# In[ ]:


#Building up datasets
l=len(list(os.listdir(CAT_TRAIN_PATH)))
index=0
lst=[]
y=[]
for file in os.listdir(CAT_TRAIN_PATH):
    index+=1
    #print(file)
    if(file=="_DS_Store"):
        continue
    im=plt.imread(CAT_TRAIN_PATH+str(file))
    im=preprocess_img(im)
    lst.append(im)
    y.append(1)
    if(index%40==0):
        print("Finished Reading "+str(index)+"/"+str(l))
        #print(im.shape)
        #print(lst.shape)
        #print(y.shape)
    
l=len(list(os.listdir(DOG_TRAIN_PATH)))
index=0
for file in os.listdir(DOG_TRAIN_PATH):
    index+=1
    #print(file)
    if(file=="_DS_Store"):
        continue
    im=plt.imread(DOG_TRAIN_PATH+str(file))
    im=preprocess_img(im)
    lst.append(im)
    y.append(0)
    if(index%40==0):
        print("Finished Reading "+str(index)+"/"+str(l))


        


# In[ ]:



print(np.array(lst).shape)
print(np.array(y).shape)


# In[ ]:


X_train=np.array(lst)
Y_train=np.array(y)
np.savez("CAT_DOG_X_train",X_train)
np.savez("CAT_DOG_Y_train",Y_train)
print(X_train.shape)
print(Y_train.shape)


# In[ ]:


#Building up datasets
l=len(list(os.listdir(CAT_TEST_PATH)))
index=0
lst=[]
y=[]
for file in os.listdir(CAT_TEST_PATH):
    index+=1
    #print(file)
    if(file=="_DS_Store"):
        continue
    im=plt.imread(CAT_TEST_PATH+str(file))
    im=preprocess_img(im)
    lst.append(im)
    y.append(1)
    if(index%40==0):
        print("Finished Reading "+str(index)+"/"+str(l))
        #print(im.shape)
        #print(lst.shape)
        #print(y.shape)
    
l=len(list(os.listdir(DOG_TEST_PATH)))
index=0
for file in os.listdir(DOG_TEST_PATH):
    index+=1
    #print(file)
    if(file=="_DS_Store"):
        continue
    im=plt.imread(DOG_TEST_PATH+str(file))
    im=preprocess_img(im)
    lst.append(im)
    y.append(0)
    if(index%40==0):
        print("Finished Reading "+str(index)+"/"+str(l))


        


# In[ ]:


X_test=np.array(lst)
Y_test=np.array(y)
print(X_test.shape)
print(Y_test.shape)
np.save("CAT_DOG_X_test",X_test)
np.save("CAT_DOG_Y_test",Y_test)


# # <div align="center">STAGE II</div>

# # Till last part we converted into numpy arrays and stored them. Now we inputted the same arrays and we would read from them

# # GETTING DATA INTO THE DIFFERENT SETS

# In[ ]:


X_TRAIN_FILE="/kaggle/input/cat-dog-numpy/CAT_DOG_X_train.npz"
X_TEST_FILE="/kaggle/input/cat-dog-numpy/CAT_DOG_X_test.npy"
Y_TRAIN_FILE="/kaggle/input/cat-dog-numpy/CAT_DOG_Y_train.npz"
Y_TEST_FILE="/kaggle/input/cat-dog-numpy/CAT_DOG_Y_test.npy"


# In[ ]:


a=np.load(X_TRAIN_FILE)
X_train=a.f.arr_0
a=np.load(Y_TRAIN_FILE)
Y_train=a.f.arr_0
a=np.load(X_TEST_FILE)
X_test=a
a=np.load(Y_TEST_FILE)
Y_test=a


# In[ ]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[ ]:


print("Distribution of cats and dogs in the different sets")
print("TRAIN  :  "+str(sum(Y_train==1))+" cats vs "+str(sum(Y_train==0))+" dogs")
print("TEST  :  "+str(sum(Y_test==1))+" cats vs "+str(sum(Y_test==0))+" dogs")


# In[ ]:


X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)


# In[ ]:


print(X_train.shape)
print(Y_train.shape)
print(X_val.shape)
print(Y_val.shape)
print(X_test.shape)
print(Y_test.shape)


# In[ ]:


print("Distribution of cats and dogs in the different sets")
print("TRAIN  :  "+str(sum(Y_train==1))+" cats vs "+str(sum(Y_train==0))+" dogs")
print("VAL  :  "+str(sum(Y_val==1))+" cats vs "+str(sum(Y_val==0))+" dogs")
print("TEST  :  "+str(sum(Y_test==1))+" cats vs "+str(sum(Y_test==0))+" dogs")


# In[ ]:



print("Images 1 to 5 :")
for i in range(0,5):
    plt.imshow(X_train[i])
    print(Y_train[i])
    plt.show()


# # UTILITY FUNCTIONS

# In[ ]:


def normalize_X(X):
    X_norm=X/255
    return X_norm


# In[ ]:


def preprocess_data(X_train,X_val,X_test,y_train,y_val,y_test):
    X_train=normalize_X(X_train)
    X_val=normalize_X(X_val)
    X_test=normalize_X(X_test)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_val, y_val = shuffle(X_val, y_val, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)
    return X_train,X_val,X_test,y_train,y_val,y_test


# In[ ]:


X_train,X_val,X_test,Y_train,Y_val,Y_test=preprocess_data(X_train,X_val,X_test,Y_train,Y_val,Y_test)


# In[ ]:




def input_and_run(model,X_train,X_val,X_test,y_train,y_val,y_test,alpha=0.01,num_epochs=10):
    
    #compile model using accuracy to measure model performance
    opt = keras.optimizers.Adam(learning_rate=alpha)
    opt2=SGD(lr=alpha, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
    
    #train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=num_epochs)
    
    #Getting results
    result = model.evaluate(X_train,y_train)
    #print(result)
    print("Training accuracy = "+str(result[1]*100))
    result = model.evaluate(X_val,y_val)
    #print(result)
    print("Validation accuracy = "+str(result[1]*100))
    result = model.evaluate(X_test,y_test)
    #print(result)
    print("Test accuracy = "+str(result[1]*100))


# In[ ]:


##BUILDING THE MODEL 1

model1 = Sequential()#add model layers

model1.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(100, 100, 3)))
model1.add(MaxPooling2D((2, 2)))
model1.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Flatten())
model1.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model1.add(Dense(1, activation='sigmoid'))

print(model1.summary())
input_and_run(model1,X_train,X_val,X_test,Y_train,Y_val,Y_test,alpha=0.0001,num_epochs=20)


# # The previous model gives us (learning_rate = 0.0001, epochs =20)
# # TRAINING ACCURACY : 99.7%
# # VALIDATION ACCURACY : 79.07%
# # TEST ACCURACY : 77.46%

# In[ ]:


##BUILDING THE MODEL 1

model2 = Sequential()#add model layers

model2.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(100, 100, 3)))
model2.add(MaxPooling2D((2, 2)))
model2.add(Dropout(0.2))
model2.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model2.add(MaxPooling2D((2, 2)))
model2.add(Dropout(0.2))
model2.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model2.add(MaxPooling2D((2, 2)))
model2.add(Dropout(0.2))
model2.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model2.add(MaxPooling2D((2, 2)))
model2.add(Dropout(0.2))
model2.add(Flatten())
model2.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model2.add(Dropout(0.5))
model2.add(Dense(1, activation='sigmoid'))

print(model2.summary())
input_and_run(model2,X_train,X_val,X_test,Y_train,Y_val,Y_test,alpha=0.002,num_epochs=200)


# In[ ]:


model2.save("model2.h5")
print("Saved model to disk")


# # The previous model gives us (learning_rate = 0.002, epochs =200)
# # TRAINING ACCURACY : 99.8%
# # VALIDATION ACCURACY : 81.8%
# # TEST ACCURACY : 83.8%

# # <div align="center">STAGE III</div>

# In[ ]:


WEIGHTS_FILE="/kaggle/input/cat-dog-numpy/model2.h5"


# In[ ]:


from keras.models import load_model
# load model
loaded_model=load_model(WEIGHTS_FILE)
print("Loaded model from disk")


# In[ ]:


#Getting results
opt = keras.optimizers.Adam(learning_rate=0.002)
loaded_model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
result = loaded_model.evaluate(X_train,Y_train)
#print(result)
print("Training accuracy = "+str(result[1]*100))
result = loaded_model.evaluate(X_val,Y_val)
#print(result)
print("Validation accuracy = "+str(result[1]*100))
result = loaded_model.evaluate(X_test,Y_test)
#print(result)
print("Test accuracy = "+str(result[1]*100))


# # <div align="center">STAGE IV</div>

# ## TRYING DATA AUGMENTATION

# In[ ]:


X_TRAIN_FILE="/kaggle/input/cat-dog-numpy/CAT_DOG_X_train.npz"
X_TEST_FILE="/kaggle/input/cat-dog-numpy/CAT_DOG_X_test.npy"
Y_TRAIN_FILE="/kaggle/input/cat-dog-numpy/CAT_DOG_Y_train.npz"
Y_TEST_FILE="/kaggle/input/cat-dog-numpy/CAT_DOG_Y_test.npy"


# In[ ]:


a=np.load(X_TRAIN_FILE)
X_train=a.f.arr_0
a=np.load(Y_TRAIN_FILE)
Y_train=a.f.arr_0
a=np.load(X_TEST_FILE)
X_test=a
a=np.load(Y_TEST_FILE)
Y_test=a

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)


# In[ ]:


print(X_train.shape)
print(Y_train.shape)
print(X_val.shape)
print(Y_val.shape)
print(X_test.shape)
print(Y_test.shape)


# In[ ]:


def normalize_X(X):
    X_norm=X/255
    return X_norm


# In[ ]:


def preprocess_data(X_train,X_val,X_test,y_train,y_val,y_test):
    X_train=normalize_X(X_train)
    X_val=normalize_X(X_val)
    X_test=normalize_X(X_test)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_val, y_val = shuffle(X_val, y_val, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)
    return X_train,X_val,X_test,y_train,y_val,y_test


# In[ ]:


X_train,X_val,X_test,Y_train,Y_val,Y_test=preprocess_data(X_train,X_val,X_test,Y_train,Y_val,Y_test)


# In[ ]:




def input_and_run2(model,X_train,X_val,X_test,y_train,y_val,y_test,alpha=0.01,num_epochs=10):
    
    datagen = ImageDataGenerator(width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)
    datagen.fit(X_train)
    
    #compile model using accuracy to measure model performance
    opt = keras.optimizers.Adam(learning_rate=alpha)
    opt2=SGD(lr=alpha, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
    
    #train the model
    model.fit(datagen.flow(X_train, y_train),validation_data=(X_val, y_val), epochs=num_epochs)
    
    #Getting results
    result = model.evaluate(X_train,y_train)
    #print(result)
    print("Training accuracy = "+str(result[1]*100))
    result = model.evaluate(X_val,y_val)
    #print(result)
    print("Validation accuracy = "+str(result[1]*100))
    result = model.evaluate(X_test,y_test)
    #print(result)
    print("Test accuracy = "+str(result[1]*100))



# In[ ]:





# In[ ]:


##BUILDING THE MODEL 1

model4 = Sequential()#add model layers

model4.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(100, 100, 3)))
model4.add(MaxPooling2D((2, 2)))
model4.add(Dropout(0.2))
model4.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model4.add(MaxPooling2D((2, 2)))
model4.add(Dropout(0.2))
model4.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model4.add(MaxPooling2D((2, 2)))
model4.add(Dropout(0.2))
model4.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model4.add(MaxPooling2D((2, 2)))
model4.add(Dropout(0.2))
model4.add(Flatten())
model4.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model4.add(Dropout(0.5))
model4.add(Dense(1, activation='sigmoid'))

print(model4.summary())
input_and_run2(model4,X_train,X_val,X_test,Y_train,Y_val,Y_test,alpha=0.001,num_epochs=200)


# ## The previous model gives us (learning_rate = 0.001, epochs = 200)(using datagen iterator)
# # TRAINING ACCURACY : 98.55%
# # VALIDATION ACCURACY : 91.45%
# # TEST ACCURACY : 92.09%

# In[ ]:


model4.save("model4.h5")
print("Saved model to disk")


# # <div align="center">STAGE V</div>

# In[ ]:


X_TRAIN_FILE="/kaggle/input/cat-dog-numpy/CAT_DOG_X_train.npz"
X_TEST_FILE="/kaggle/input/cat-dog-numpy/CAT_DOG_X_test.npy"
Y_TRAIN_FILE="/kaggle/input/cat-dog-numpy/CAT_DOG_Y_train.npz"
Y_TEST_FILE="/kaggle/input/cat-dog-numpy/CAT_DOG_Y_test.npy"


# In[ ]:


a=np.load(X_TRAIN_FILE)
X_train=a.f.arr_0
a=np.load(Y_TRAIN_FILE)
Y_train=a.f.arr_0
a=np.load(X_TEST_FILE)
X_test=a
a=np.load(Y_TEST_FILE)
Y_test=a

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)


# In[ ]:


print(X_train.shape)
print(Y_train.shape)
print(X_val.shape)
print(Y_val.shape)
print(X_test.shape)
print(Y_test.shape)


# In[ ]:


def normalize_X(X):
    X_norm=X/255
    return X_norm


# In[ ]:


def preprocess_data(X_train,X_val,X_test,y_train,y_val,y_test):
    X_train=normalize_X(X_train)
    X_val=normalize_X(X_val)
    X_test=normalize_X(X_test)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_val, y_val = shuffle(X_val, y_val, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)
    return X_train,X_val,X_test,y_train,y_val,y_test


# In[ ]:


X_train,X_val,X_test,Y_train,Y_val,Y_test=preprocess_data(X_train,X_val,X_test,Y_train,Y_val,Y_test)


# In[ ]:




def input_and_run3(model,X_train,X_val,X_test,y_train,y_val,y_test,alpha=0.01,num_epochs=10):
    
    #datagen = ImageDataGenerator(width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)
    #datagen.fit(X_train)
    
    #compile model using accuracy to measure model performance
    opt = keras.optimizers.Adam(learning_rate=alpha)
    opt2=SGD(lr=alpha, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
    
    #train the model
    model.fit(X_train,y_train,validation_data=(X_val, y_val), epochs=num_epochs)
    #model.fit(datagen.flow(X_train, y_train),validation_data=(X_val, y_val), epochs=num_epochs)
    
    #Getting results
    result = model.evaluate(X_train,y_train)
    #print(result)
    print("Training accuracy = "+str(result[1]*100))
    result = model.evaluate(X_val,y_val)
    #print(result)
    print("Validation accuracy = "+str(result[1]*100))
    result = model.evaluate(X_test,y_test)
    #print(result)
    print("Test accuracy = "+str(result[1]*100))



# In[ ]:


model5 = Sequential()
model5.add(VGG16(include_top=False, input_shape=(100, 100, 3)))
# mark loaded layers as not trainable
for layer in model5.layers:
    layer.trainable = False
# add new classifier layers
#flat1 = Flatten()(model.layers[-1].output)
#class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
#output = Dense(1, activation='sigmoid')(class1)

model5.add(Flatten())
model5.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model5.add(Dropout(0.5))
model5.add(Dense(1, activation='sigmoid'))
print(model5.summary())
input_and_run3(model5,X_train,X_val,X_test,Y_train,Y_val,Y_test,alpha=0.001,num_epochs=200)


# define new model
#model = Model(inputs=model.inputs, outputs=output)
# compile model
#opt = SGD(lr=0.001, momentum=0.9)
#model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
#return model


# # The previous model gives us (learning_rate = 0.001, epochs = 200)
# 
# # Accuracy in first epoch was 80% 
# 
# # TRAINING ACCURACY : 100%
# # VALIDATION ACCURACY : 86.76%
# # TEST ACCURACY : 87.09%

# In[ ]:


model5.save("model5.h5")
print("Saved model to disk")


# # <div align="center">STAGE VI</div>

# # DATA AUGMENTATION + TRANSFER LEARNING

# In[ ]:


X_TRAIN_FILE="/kaggle/input/cat-dog-numpy/CAT_DOG_X_train.npz"
X_TEST_FILE="/kaggle/input/cat-dog-numpy/CAT_DOG_X_test.npy"
Y_TRAIN_FILE="/kaggle/input/cat-dog-numpy/CAT_DOG_Y_train.npz"
Y_TEST_FILE="/kaggle/input/cat-dog-numpy/CAT_DOG_Y_test.npy"


# In[ ]:


a=np.load(X_TRAIN_FILE)
X_train=a.f.arr_0
a=np.load(Y_TRAIN_FILE)
Y_train=a.f.arr_0
a=np.load(X_TEST_FILE)
X_test=a
a=np.load(Y_TEST_FILE)
Y_test=a

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)


# In[ ]:


print(X_train.shape)
print(Y_train.shape)
print(X_val.shape)
print(Y_val.shape)
print(X_test.shape)
print(Y_test.shape)


# In[ ]:


def normalize_X(X):
    X_norm=X/255
    return X_norm


# In[ ]:


def preprocess_data(X_train,X_val,X_test,y_train,y_val,y_test):
    X_train=normalize_X(X_train)
    X_val=normalize_X(X_val)
    X_test=normalize_X(X_test)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_val, y_val = shuffle(X_val, y_val, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)
    return X_train,X_val,X_test,y_train,y_val,y_test


# In[ ]:


X_train,X_val,X_test,Y_train,Y_val,Y_test=preprocess_data(X_train,X_val,X_test,Y_train,Y_val,Y_test)


# In[ ]:




def input_and_run4(model,X_train,X_val,X_test,y_train,y_val,y_test,alpha=0.01,num_epochs=10):
    
    datagen = ImageDataGenerator(width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)
    datagen.fit(X_train)
    
    #compile model using accuracy to measure model performance
    opt = keras.optimizers.Adam(learning_rate=alpha)
    opt2=SGD(lr=alpha, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
    
    #train the model
    #model.fit(X_train,y_train,validation_data=(X_val, y_val), epochs=num_epochs)
    model.fit(datagen.flow(X_train, y_train),validation_data=(X_val, y_val), epochs=num_epochs)
    
    #Getting results
    result = model.evaluate(X_train,y_train)
    #print(result)
    print("Training accuracy = "+str(result[1]*100))
    result = model.evaluate(X_val,y_val)
    #print(result)
    print("Validation accuracy = "+str(result[1]*100))
    result = model.evaluate(X_test,y_test)
    #print(result)
    print("Test accuracy = "+str(result[1]*100))



# In[ ]:


model6 = Sequential()
model6.add(VGG16(include_top=False, input_shape=(100, 100, 3)))
# mark loaded layers as not trainable
for layer in model6.layers:
    layer.trainable = False
# add new classifier layers
#flat1 = Flatten()(model.layers[-1].output)
#class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
#output = Dense(1, activation='sigmoid')(class1)

model6.add(Flatten())
model6.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model6.add(Dropout(0.5))
model6.add(Dense(1, activation='sigmoid'))
print(model6.summary())
input_and_run4(model6,X_train,X_val,X_test,Y_train,Y_val,Y_test,alpha=0.001,num_epochs=200)


# define new model
#model = Model(inputs=model.inputs, outputs=output)
# compile model
#opt = SGD(lr=0.001, momentum=0.9)
#model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
#return model


# # The previous model gives us (learning_rate = 0.001, epochs = 200)
# 
# # Accuracy in first epoch was 80% 
# 
# # TRAINING ACCURACY : 97.03%
# # VALIDATION ACCURACY : 87.81%
# # TEST ACCURACY : 87.83%

# In[ ]:


model6.save("model6.h5")
print("Saved model to disk")


# In[ ]:


a=4


# In[ ]:




